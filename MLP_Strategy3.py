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
from LossFunction import FrobeniusPenalty, NegativeRankICLoss, NegativeICLoss
from numpy.lib.stride_tricks import sliding_window_view
import xgboost as xgb
import sys
from MLP import NN
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
        dfs = pd.concat(all_dfs, ignore_index=True)
        dfs['T'] = pd.to_datetime(dfs['T']).dt.date
        return dfs
    else:
        return pd.DataFrame()

target_name = 'Return5'


# features = ['CP', 'OP', 'LP', 'HP', 'Amount', 'Volume', 'VWAPRe',
# 'BAmountL', 'BAmountM', 'BAmountS', 'BAmountXL', 'EP', 'SAmountL', 'SAmountM', 'SAmountXL', 'Turnover',
# 'L2BAmount', 'L2BMoney', 'L2BRate', 'L2LBMoney', 'L2LBRate', 'L2LSAmount', 'L2LSMoney', 'L2LSRate',
# 'L2SAmount', 'L2SMoney', 'L2SRate']

# features = ['CP', 'OP', 'LP', 'HP', 'Amount', 'Volume', 'VWAPRe',
# 'BAmountL', 'BAmountM', 'BAmountS', 'BAmountXL', 'EP', 'SAmountL', 'SAmountM', 'SAmountXL', 'Turnover']

features = ['CP', 'OP', 'LP', 'HP', 'Amount', 'Volume', 'VWAPRe']

## 模型参数
batch_size = 1000
clip_value = 3
stopping_patience = 10
input_size = len(features)
hidden_size = [64]
output_size = 1
factor_num = output_size
num_epochs = 100
test_result = pd.DataFrame()

## 扩展窗口训练
train_year=2008
for i in range(11):
    train_start, train_end = dt.date(train_year, 1, 1), dt.date(train_year+3+i, 12, 31)
    val_start, val_end = dt.date(train_year+4+i, 1, 1), dt.date(train_year+4+i, 12, 31)
    test_start, test_end = dt.date(train_year+5+i, 1, 1), dt.date(train_year+5+i, 12, 31)
    train_df = getData(train_start, train_end, "/home/lenovo/liangluoming/results/DailyTradeData+L2Data")
    val_df = getData(val_start, val_end, "/home/lenovo/liangluoming/results/DailyTradeData+L2Data")
    test_df = getData(test_start, test_end, "/home/lenovo/liangluoming/results/DailyTradeData+L2Data")
    print("训练区间:{}-{}, 验证区间:{}-{}, 测试区间:{}-{}".format(str(train_start), str(train_end), str(val_start), str(val_end), str(test_start), str(test_end)))
    for feature in features:
        train_df = train_df[train_df[feature].notnull()]
        val_df = val_df[val_df[feature].notnull()]
        test_df = test_df[test_df[feature].notnull()]

    train_df.sort_values(by=['T', 'Sk'], ignore_index=True, inplace=True)
    val_df.sort_values(by=['T', 'Sk'], ignore_index=True, inplace=True)
    test_df.sort_values(by=['T', 'Sk'], ignore_index=True, inplace=True)
    

    x_train, y_train = train_df[features].values, train_df[target_name].values.reshape((-1, 1))
    x_val, y_val = val_df[features].values, val_df[target_name].values.reshape((-1, 1))
    x_test, y_test = test_df[features].values, test_df[target_name].values.reshape((-1, 1))

    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_val = torch.FloatTensor(x_val)
    y_val = torch.FloatTensor(y_val)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    early_stopping = EarlyStopping(stopping_patience, verbose=True, save=False)

    nn_model = NN(input_size, hidden_size, output_size)

    criterion1 = nn.MSELoss()
    criterion2 = NegativeICLoss()
    val_criterion = NegativeRankICLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("成功调用GPU:{}".format(torch.cuda.get_device_name(0)))
    else:
        print("调用GPU失败:{}".format(device.type))
    nn_model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        nn_model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            factors= nn_model(batch_X)
         
            loss = criterion1(factors, batch_y)


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), clip_value)
            optimizer.step()
    
            total_loss += loss.item()


            
        average_loss = total_loss / len(train_loader)
        end_time = time.time()
        train_time = end_time - start_time


        start_time = time.time()
        with torch.no_grad():
            val_loss = 0.0
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                factors= nn_model(batch_X)

                loss =  criterion2(factors, batch_y) 


                val_loss += loss.item()
        val_loss /= len(val_loader)
        end_time = time.time()
        val_time = end_time - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, 用时: {round(train_time, 2)}s, 学习率: {optimizer.param_groups[0]['lr']}, Val_Loss: {val_loss}, 用时: {round(val_time, 2)}s")

        early_stopping(val_loss, nn_model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break
        
    outputs = []
    nn_model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = nn_model(batch_X)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).cpu().data.numpy()
    outputs = pd.DataFrame(outputs)
    outputs.columns = ['mlp_factor_{}'.format(i) for i in range(1, output_size+1)]
    test_df.reset_index(drop=True,inplace=True)
    test_df = pd.concat([test_df, outputs], axis=1)

    test_result = pd.concat([test_result, test_df])
    
test_result.to_parquet("/home/lenovo/liangluoming/results/llm_mlp_1_fix5_plus.parquet",index=False)