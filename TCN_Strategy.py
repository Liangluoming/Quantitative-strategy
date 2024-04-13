import numpy as np
import pandas as pd
import os
import warnings
import time 
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from utils import EarlyStopping, LRScheduler
from LossFunction import NegativeICLoss, NegativeRankICLoss
from numpy.lib.stride_tricks import sliding_window_view
import sys
from TCN import TemporalConvNet
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + dt.timedelta(n)

def getData(start_date, end_date, directory):
    all_dfs = []
    for single_date in daterange(start_date, end_date):
        file_name = single_date.strftime("%Y%m%d") + ".parquet"
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            df = pq.read_table(file_path).to_pandas()
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

## 预测目标
target_name = 'Return5'

## 输入变量

features = ['CP', 'OP', 'LP', 'HP', 'Amount', 'Volume', 'VWAPRe',
'BAmountL', 'BAmountM', 'BAmountS', 'BAmountXL', 'EP', 'SAmountL', 'SAmountM', 'SAmountXL', 'Turnover']

# features = ['CP', 'OP', 'LP', 'HP', 'Amount', 'Volume', 'VWAPRe',
# 'BAmountL', 'BAmountM', 'BAmountS', 'BAmountXL', 'EP', 'SAmountL', 'SAmountM', 'SAmountXL', 'Turnover',
# 'L2BAmount', 'L2BMoney', 'L2BRate', 'L2LBMoney', 'L2LBRate', 'L2LSAmount', 'L2LSMoney', 'L2LSRate',
# 'L2SAmount', 'L2SMoney', 'L2SRate']

## 模型参数
clip_value = 3
block_num = 5
num_inputs = len(features)
num_channels = [64, 64, 64, 64, 64]
kernel_size = 2
stopping_patience = 10
batch_size = 1000
time_window = 63
features_num = len(features)
factor_num = 1
dropout = 0.1


## 滚动训练
test_result = []

for test_year in range(2013, 2024, 1):
    ## 读入数据
    train_start, train_end = dt.date(test_year-5, 1, 1), dt.date(test_year-2, 12, 31)
    val_start, val_end = dt.date(test_year-1, 1, 1), dt.date(test_year-1, 12, 31)
    test_start, test_end = dt.date(test_year, 1, 1), dt.date(test_year, 12, 31)
    train_df = getData(train_start, train_end, "/home/lenovo/liangluoming/results/DailyTradeData+L2Data")
    val_df = getData(val_start, val_end, "/home/lenovo/liangluoming/results/DailyTradeData+L2Data")
    test_df = getData(test_start, test_end, "/home/lenovo/liangluoming/results/DailyTradeData+L2Data")
    df = pd.concat([train_df, val_df, test_df])
    
    ## 去除缺失值
    for feature in features:
        df = df[df[feature].notnull()]
    df['T'] = pd.to_datetime(df['T']).dt.date
    ## 按照交易日期排序
    df.sort_values(by=['T', 'Sk'], ignore_index=True, inplace=True)
    del train_df
    del val_df
    del test_df

    ## 构建输入
    df = df[df.groupby('Sk').CP.transform(lambda x : len(x))>time_window]
    df.reset_index(drop=True, inplace=True)

    X = np.concatenate(df.groupby(['Sk']).apply(lambda x:sliding_window_view(x[features], window_shape=(time_window, features_num))).values).reshape((-1,features_num, time_window))
    df_numpy = np.concatenate(df.groupby(['Sk']).apply(lambda x:sliding_window_view(x[['Sk','T', target_name]], window_shape=(time_window, 3))).values).reshape((-1, time_window, 3))
    df_numpy = pd.DataFrame(df_numpy[:,-1,:])
    df_numpy.columns = ['Sk', 'T', target_name]
    df_numpy[target_name] = df_numpy[target_name].astype(float)


    print("=====样本划分=====")
    print("训练区间:{}-{}, 验证区间:{}-{}, 测试区间:{}-{}".format(str(dt.date(test_year-5, 1, 1)), str(dt.date(test_year-2, 12, 31)), str(dt.date(test_year-1, 1, 1)), str(dt.date(test_year-1, 12, 31)), str(dt.date(test_year, 1, 1)), str(dt.date(test_year, 12, 31))))
    train_idx = df_numpy[(df_numpy['T']>=dt.date(test_year-5, 1, 1)) & (df_numpy['T']<=dt.date(test_year-2, 12, 31))].index
    val_idx = df_numpy[(df_numpy['T']>=dt.date(test_year-1, 1, 1)) & (df_numpy['T']<=dt.date(test_year-1, 12, 31))].index
    test_idx = df_numpy[(df_numpy['T']>=dt.date(test_year, 1, 1)) & (df_numpy['T']<=dt.date(test_year, 12, 31))].index

    x_train, y_train = X[train_idx,:,:], df_numpy.loc[train_idx, target_name].values.reshape((-1, 1))
    x_val, y_val = X[val_idx,:,:], df_numpy.loc[val_idx, target_name].values.reshape((-1, 1))
    x_test, y_test = X[test_idx,:,:], df_numpy.loc[test_idx,target_name].values.reshape((-1, 1))


    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_val = torch.FloatTensor(x_val)
    y_val = torch.FloatTensor(y_val)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ## 设置早停机制
    early_stopping = EarlyStopping(stopping_patience, verbose=True, save=False)

    ## 调用TCN模型
    tcn_model = TemporalConvNet(num_inputs, num_channels, factor_num, kernel_size=2, dropout=dropout)
    
    criterion1 = nn.MSELoss() ## MSE损失函数
    criterion2 = NegativeICLoss()  ## pearson IC 损失函数
    criterion3 = NegativeRankICLoss() ## Rank IC 损失函数
    optimizer = optim.Adam(tcn_model.parameters(), lr=0.0001)

    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("成功调用GPU:{}".format(torch.cuda.get_device_name(0)))
    else:
        print("调用GPU失败:{}".format(device.type))
    tcn_model.to(device)

    ## 训练集
    for epoch in range(num_epochs):
        start_time = time.time()
        tcn_model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            factors = tcn_model(batch_X)
  
            loss = criterion1(factors, batch_y) 

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn_model.parameters(), clip_value)
            optimizer.step()
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        end_time = time.time()
        train_time = end_time - start_time

        ## 验证集
        start_time = time.time()
        with torch.no_grad():
            val_loss = 0.0
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
   
                factors = tcn_model(batch_X)

                loss = criterion2(factors, batch_y) 

                val_loss += loss.item()
        val_loss /= len(val_loader)
        end_time = time.time()
        val_time = end_time - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, 用时: {round(train_time, 2)}s, 学习率: {optimizer.param_groups[0]['lr']}, Val_Loss: {val_loss}, 用时: {round(val_time, 2)}s")

        ## 早停判断
        early_stopping(val_loss, tcn_model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break

    ## 测试集
    outputs = []
    tcn_model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = tcn_model(batch_X)
            
            outputs.append(output)
    outputs = torch.cat(outputs).cpu().data.numpy()
    test_df = df_numpy[(df_numpy['T']>=dt.date(test_year, 1, 1)) & (df_numpy['T']<=dt.date(test_year, 12, 31))]
    outputs = pd.DataFrame(outputs)
    outputs.columns = ['tcn{}'.format(i) for i in range(1, factor_num+1)]
    test_df.reset_index(drop=True,inplace=True)
    test_df = pd.concat([test_df, outputs], axis=1)
    test_df = pd.merge(df[['Sk', 'T', 'Return','ResidualR']], test_df, on=['Sk', 'T'])
    test_result.append(test_df)
test_result = pd.concat(test_result)

test_result.to_parquet("/home/lenovo/liangluoming/results/DailyTradeData+L2Data_TCN2.parquet",index=False)