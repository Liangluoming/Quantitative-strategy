"""
@author: LiangLuoming
@email: liangluoming00@163.com
@github: https://github.com/Liangluoming
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import datetime as dt
import warnings
from tqdm import tqdm
import time
import xgboost as xgb
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor

import os
os.chdir("~")

def get_paths(path,*suffix):
    # 获取读取文件的路径
    # path: 文件路径
    # *suffix: 文件后缀 
    pathArray = []
    for r,ds,fs in os.walk(path):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r,fn)
                pathArray.append(fname)
    return pathArray 

def train_model(y_train, x_train, n_estimators = 200, criterion = 'squared_error', random_state = 123):
    """
    训练机器学习模型
    """
    # max_depth = int(len(features) / 3)
    # xgb_model = xgb.XGBRegressor(max_depth = max_depth, n_estimators = n_estimators, learning_rate = learning_rate, device = 'cuda')
    # xgb_model.fit(x_train, y_train)
    rf_model = RandomForestRegressor(n_estimators = n_estimators, criterion = criterion, random_state = random_state, n_jobs = 8, warm_start = True)
    rf_model.fit(x_train, y_train)

    return rf_model

def get_cum_ret(ret):
    # 累计收益率
    cum_ret = (ret + 1).prod() - 1
    return cum_ret

def get_annual_ret(ret, freq):
    # 年化收益率
    annual_ret = (ret + 1).prod() ** (freq / len(ret)) - 1
    return annual_ret

def get_max_drawdownRate(ret):
    # 最大回撤
    ret = (ret + 1).cumprod()
    endDate = np.argmax((np.maximum.accumulate(ret)) - ret)
    if endDate == 0:
        return 0, len(ret), endDate
    else:
        startDate = np.argmax(ret[:endDate])
    return (ret[startDate] - ret[endDate]) / ret[startDate], startDate, endDate


def strategy(model, data, test_features, test_sign, liqudity_factor, return_features):
    """
    策略
    Args:
        model: 机器学习模型
        data: 数据集
        test_features: 因子列
        test_sign: 择时信号
        liqudity_factor: 流动性因子
        return_features: 收益率
    Returns:
        long_index: 做多股票索引
        short_index: 做空股票索引
    """
    df = data.copy()
    df = df[df[test_sign] == 1] ## 只考虑开仓信号的股票
    df[return_features] = model.predict(df[test_features]) ## 用机器学习模型预测收益率
    df['score'] = df[return_features] * df[liqudity_factor] ## 收益率 * 流动性因子得到得分
    df.sort_values(by = 'score', ascending = False, inplace = True) ## 得分从高到低排
    long = df[df[return_features]>0.003] ## 做多的股票需要保证预测收益率不低于0.3%
    short =  df[df[return_features]< -0.003] ## 做空的股票保证预测收益率不高于-0.3%
    if len(long):
        long_index = long.index[:min(len(long),10)] ## 取前10只股票
    else:
        long_index = [] 
    if len(short):
        short_index = short.index[- min(len(short),10):] ## 取后10只股票
    else:
        short_index = []
    return long_index, short_index

def liqudity_sign(data, col):
    """
    股票择时信号
    Args:
        col: 流动性因子
    """
    df = data.copy()
    ## 计算过去5个交易日流动性因子的均值和标准差
    mean, std = df[col].rolling(window = 6, closed = 'left').mean(), df[col].rolling(window = 6, closed = 'left').std()

    df['sign_high'] = mean + 2 * std ## 设置流动性上限
    df['sign_low'] = mean - std  ## 设置流动性下限
    indicies = df[df['sign_high'].isnull()].index.to_numpy() ## 不足5个交易日（无法计算流动性因子）的都设置为开仓信号
    df['sign'] = ((df[col] <= df['sign_high']) & (df[col] >= df['sign_low'])).astype('int') ##流动性在范围内开仓
    df.loc[indicies, 'sign'] = 1
    return df['sign']

def get_T1return(data, column):
    """
    计算收益率
    """
    return data[column].shift(-1) / data[column] - 1

def get_cum_ret(ret):
    # 累计收益率
    cum_ret = (ret + 1).prod() - 1
    return cum_ret

def get_annual_ret(ret, freq):
    # 年化收益率
    annual_ret = (ret + 1).prod() ** (freq / len(ret)) - 1
    return annual_ret


def get_annual_vol(ret, freq):
    # 年化波动率
    annual_vol = np.sqrt(freq) * ret.std()
    return annual_vol

def get_sharpe_ratio(ret, rf, freq):
    # 夏普比率
    sharpe = (get_annual_ret(ret, freq) - rf) / get_annual_vol(ret, freq)
    return sharpe
    
def get_max_drawdownRate(ret):
    # 最大回撤
    ret = (ret + 1).cumprod()
    endDate = np.argmax((np.maximum.accumulate(ret)) - ret)
    if endDate == 0:
        return 0, len(ret), endDate
    else:
        startDate = np.argmax(ret[:endDate])
    return (ret[startDate] - ret[endDate]) / ret[startDate], startDate, endDate

# def get_max_drawdownRate(ret):
#     # 最大回撤率
#     drawdown, startDate, _ = get_max_drawdown(ret)
#     return drawdown / ret[startDate]

def get_calmarRatio(ret, freq):
    # Calmar比率
    maxdownRate, _, __ = get_max_drawdownRate(ret)
    return get_annual_ret(ret, freq) / maxdownRate

def get_winRate(ret, sign):
    """
    计算胜率
    """
    return np.sum(ret[sign == 1] > 0) / len(ret[sign == 1])

def get_ProfitLossRatio(ret):
    """
    计算盈亏比
    """
    profit = np.diff(np.append([1], (1 + ret).cumprod()))
    return profit[profit > 0].sum() / (- profit[profit < 0].sum())

def get_BetaAlpha(ret, bench_ret, rf, freq):
    """
    计算 beta alpha
    """
    beta = np.cov(ret, bench_ret)[0, 1] / np.var(bench_ret)
    alpha = get_annual_ret(ret, freq) - (rf + beta * (get_annual_ret(bench_ret, freq) - rf))
    return beta, alpha

def get_InformationRatio(ret, bench_ret, freq):
    """
    计算IR
    """
    return (get_annual_ret(ret, freq) - get_annual_ret(bench_ret, freq)) / get_annual_vol(ret - bench_ret, freq)

def evaluate(df, column, sign_column, bench_ret, rf, freq):
    """
    投资组合表现评价
    """
    data = df.copy()
    # data['factual_ret'] = data['Retindex'] * data['signal']
    print('回测期间: %s-%s' % (str(data['Trddt'][0]).replace('-', '/'), str(data['Trddt'].values[-1]).replace('-', '/')))
    print('总收益率: 策略: %.2f%%' % (get_cum_ret(data[column]) * 100))
    print('年化收益率: 策略: %.2f%%' % (get_annual_ret(data[column], 252) * 100))
    print('年化波动率: 策略: %.2f%%' % (get_annual_vol(data[column], 252) * 100))
    print('夏普比率: 策略: %.2f' % (get_sharpe_ratio(data[column], rf, 252)))
    strategy_maxdown, startDate, endDate = get_max_drawdownRate(data[column])
    print('最大回撤: 策略: %.2f%%' % (strategy_maxdown * 100))
    print('策略最大回撤开始时间：%s, 最大回撤结束时间：%s' % (data['Trddt'][startDate], data['Trddt'][endDate]))
    print('Calmar 比率: 策略: %.2f' %(get_calmarRatio(data[column], 252)))
    print("盈亏比: %.2f" %(get_ProfitLossRatio(data[column])))
    if sign_column != '':
        print("胜率: %.2f%%" %(get_winRate(data[column], data[sign_column]) * 100))
    
    if len(bench_ret):
        beta, alpha = get_BetaAlpha(df[column], bench_ret, rf, freq)
        print("beta: %.2f, alpha: %.2f" %(beta, alpha))
        print("IR: %.2f%%" %(get_InformationRatio(data[column], bench_ret, freq) * 100))


## 因子数据路径
factor_paths = get_paths("./tidy_data", '.csv')
factor_paths.sort()

## 交易数据路径
trade_paths = get_paths("./股票时序交易数据", '.csv')
trade_paths.sort()

## 保留的因子
features = [
    'alpha163', 'alpha101', 'alpha155', 'alpha145',
       'alpha084', 'alpha070', 'alpha108', 'alpha002', 'alpha085', 'alpha144',
       'alpha154', 'alpha119', 'alpha061', 'alpha013', 'alpha025', 'alpha134',
       'alpha124', 'alpha179', 'alpha001', 'alpha073', 'alpha102', 'alpha170',
       'alpha008', 'alpha044', 'alpha036', 'alpha103', 'alpha113', 'alpha125',
       'alpha178', 'alpha168', 'alpha062', 'alpha005', 'alpha077', 'alpha142',
       'alpha048', 'alpha130', 'alpha120', 'alpha180', 'alpha041', 'alpha033',
       'alpha138', 'alpha040', 'alpha032', 'alpha004', 'alpha076', 'alpha092',
       'alpha131', 'alpha042', 'alpha074', 'alpha064', 'alpha016', 'alpha080',
       'alpha029', 'alpha141', 'alpha007', 'alpha017', 'alpha091', 'alpha104',
       'alpha114', 'alpha043'
]
## 剔除共线性特征
drop_alphas = ['alpha026', 'alpha094', 'alpha081', 'alpha095', 'alpha097', 'alpha100', 'alpha132']


## 读入数据
factors = []
tradings = []
for i in tqdm(range(len(factor_paths))):
    factors.append(pd.read_csv(factor_paths[i], encoding = 'gbk'))
    tradings.append(pd.read_csv(trade_paths[i]))
factors = pd.concat(factors)
factors.drop(columns = drop_alphas, inplace = True)
tradings = pd.concat(tradings)
tradings['F_Dretwd'].fillna(0, inplace = True)
tradings.sort_values(by = ['Stkcd', 'Trddt'], ascending = [True, True], ignore_index=True, inplace = True)

## 开盘价买入开盘价卖出收益率
tradings['T1return'] = tradings.groupby('Stkcd').apply(get_T1return, 'Opnprc').droplevel(0)
tradings_pivot = tradings.pivot(index = 'Trddt', columns = 'Stkcd', values = 'T1return').shift(-1)
tradings_pivot = tradings_pivot.unstack().reset_index().rename(columns = {0 : 'F_T1return'})
tradings = pd.merge(tradings, tradings_pivot, on = ['Stkcd', 'Trddt'], how = 'left')
tradings['F_T1return'].fillna(0, inplace = True)

## 合并交易数据和因子数据
data = pd.merge(factors, tradings[['Stkcd', 'Trddt', 'liquidity', 'T1return','F_T1return']], on = ['Stkcd', 'Trddt'], how = 'left')
data['Stkcd'] = data['Stkcd'].apply(lambda x : str(x).zfill(6))
data.reset_index(drop = True, inplace = True)
## 获取股票开仓信号
data['sign'] = data.groupby('Stkcd').apply(liqudity_sign, 'liquidity').droplevel(0)

## 读入上市时间数据
list_time = pd.read_csv("./raw_data/TRD_Co.csv")
list_time['Stkcd'] = list_time['Stkcd'].apply(lambda x : str(x).zfill(6))
list_time['Listdt'] = pd.to_datetime(list_time['Listdt']).dt.date

##合并数据
data = pd.merge(data, list_time, on ='Stkcd', how = 'left')
data['Trddt'] = pd.to_datetime(data['Trddt']).dt.date
## 交易日距离上市时间的间隔
data['list_time'] = (data['Trddt'] - data['Listdt']).apply(lambda x : x.days)
data = data[data['list_time']>365].reset_index(drop = True) ## 去除上市不满一年的数据

### 市场流动性择时信号

## 计算市值占比
data['marketRatio'] = data.Dsmvosd / data.groupby('Trddt').Dsmvosd.transform(lambda x: np.sum(x))
data['weight_Dretwd'] = data['marketRatio'] * data['Dretwd'] ## 加权收益率1
data['weight_T1return'] = data['marketRatio'] * data['T1return'] ## 加权收益率2
market_liquidity = data[['Trddt', 'weight_Dretwd', 'weight_T1return', 'liquidity']].groupby('Trddt').agg({'weight_Dretwd':np.sum, 'weight_T1return':np.sum, 'liquidity' : np.mean}).reset_index()

market_liquidity['mean_rolling'] = market_liquidity['liquidity'].rolling(window = 6, closed = 'left').mean() ## 计算过去5个交易日的流动性因子均值
market_liquidity['std_rolling'] = market_liquidity['liquidity'].rolling(window = 6, closed = 'left').std() ## 计算过去5个交易日的流动性因子方差
market_liquidity['Dretwd_mean_rolling'] = market_liquidity['weight_Dretwd'].rolling(window = 6, closed = 'left').mean() ## 计算加权收益率2过去5个交易日的均值
## 满足特定条件方可发出开仓信号
market_liquidity['market_sign'] = (((market_liquidity['weight_Dretwd'] < market_liquidity['mean_rolling'] + 2 * market_liquidity['std_rolling']) | (market_liquidity['weight_Dretwd'] > market_liquidity['Dretwd_mean_rolling'])) & (market_liquidity['liquidity'] > market_liquidity['mean_rolling'] - 1 * market_liquidity['std_rolling'])).astype('int')
## 无法计算开仓信号的设为开仓
market_liquidity.loc[market_liquidity[market_liquidity['mean_rolling'].isnull()].index, 'market_sign'] = 1

## 获取每个交易节点, 提高遍历速度
indicies = data.drop_duplicates(subset = 'Trddt',keep = 'last').index.to_numpy() + 1
dtime = data['Trddt'].unique()

long_series = [] ## 用来存放做多组合的收益率序列
long_short_series = [] ## 用来存放多空组合的收益率序列
return_features = 'F_Dretwd' ## F_T1return
for i in tqdm(range(7, len(indicies) - 1)):
    ## 如果市场择时信号为空仓则跳过
    if market_liquidity[market_liquidity['Trddt'] == dtime[i]]['market_sign'].values[0] == 0:
        long_series.append(0)
        long_short_series.append(0)
        continue
    ## 过去15个交易日的数据作为训练集
    y_train = data.loc[indicies[max(0, i - 15)] : indicies[i-1], return_features]
    x_train = data.loc[indicies[max(0, i - 15)] : indicies[i-1], features]
    rf_model = train_model(y_train, x_train, n_estimators = 50) ## 训练机器学习模型

    ## 执行策略
    long_index, short_index = strategy(rf_model, data.iloc[indicies[i-1] : indicies[i], :], features, 'sign', 'liquidity', return_features)

    ## 根据策略返回的买卖股票池计算收益率
    ## 采取流动性加权
    if len(long_index):
        NAV_long = np.sum(data.loc[long_index, return_features] * data.loc[long_index, 'liquidity'] / np.sum(data.loc[long_index, 'liquidity']))
    else:
        NAV_long = 0
    if len(short_index):
        weights = np.sum(data.loc[long_index, 'liquidity']) / np.sum(data.loc[np.append(long_index,short_index), 'liquidity'])
        short_income = np.sum(data.loc[short_index, return_features] * data.loc[short_index, 'liquidity'] / np.sum(data.loc[short_index, 'liquidity']))
        if len(long_index):
            long_income = np.sum(data.loc[long_index, return_features] * data.loc[long_index, 'liquidity'] / np.sum(data.loc[long_index, 'liquidity']))
        NAV_long_short = long_income * weights - short_income * (1 - weights)
    else:
        NAV_long_short = NAV_long.copy()
    
    long_series.append(NAV_long)
    long_short_series.append(NAV_long_short)

## 读入指数日度数据
trd_index = pd.read_csv("/Users/luomingliang/Desktop/梁洛铭/研究生/UIBE/研一/Python与金融量化/期末大作业/数据/raw_data/TRD_Index.csv")
trd_index['Indexcd'] = trd_index['Indexcd'].apply(lambda x : str(x).zfill(6))
trd_index = trd_index[trd_index['Indexcd'] == '000300'] ## 选取沪深300数据
trd_index['Trddt'] = pd.to_datetime(trd_index['Trddt']).dt.date
trd_index = trd_index[trd_index['Trddt'] >= dt.date(2015,1, 5)] ## 选取起始时间
trd_index.sort_values(by = 'Trddt', ignore_index = True, inplace = True)

## 绘制净值走势图
plt.figure(figsize = (16, 4))
plt.plot(dtime[7:-1], (1+np.array(long_series)).cumprod(), label = '做多')
plt.plot(dtime[7:-1], (1+np.array(long_short_series)).cumprod(), label = '多空组合')
plt.plot(dtime[7:-1], (1 + trd_index['Retindex'][: -1]).cumprod(), label = '沪深300')
plt.legend()
plt.ylabel('累计净值')
#plt.savefig("基础策略累计净值.jpg", dpi = 1000)

## 保存数据
strategy_return = pd.DataFrame({'Trddt' : dtime[7:-1], 'Long': long_series, 'Long_Short' : long_short_series})
strategy_return = pd.merge(strategy_return, market_liquidity[['Trddt', 'market_sign']], on = 'Trddt', how = 'left')
strategy_return = pd.merge(strategy_return, trd_index[['Trddt', 'Retindex']], on = 'Trddt', how = 'left')
# strategy_return.to_csv("无交易成本买卖前10后10只股票流动性加强.csv", index = False)


## 基础策略收益率序列数据
basic_result = pd.read_csv("./无交易成本买卖前10后10只股票.csv") 
plt.figure(figsize = (16, 8))
plt.plot(pd.to_datetime(basic_result.Trddt).dt.date, (1 + basic_result['Long']).cumprod(), label = '做多')
plt.plot(pd.to_datetime(basic_result.Trddt).dt.date, (1 + basic_result['Long_Short']).cumprod(), label = '多空组合')
plt.plot(pd.to_datetime(basic_result.Trddt).dt.date, (1 + basic_result['Retindex']).cumprod(), label = '沪深300')
plt.legend()
plt.ylabel('累计净值')
# plt.savefig("图1基础策略累计净值.jpg", dpi = 1000)

## 基准策略评估
evaluate(basic_result, 'Long', 'market_sign', basic_result['Retindex'], 0.03, 252)

evaluate(basic_result, 'Long_Short', 'market_sign', basic_result['Retindex'], 0.03, 252)

## 基准策略交易成本分析
plt.figure(figsize = (16, 8))
for tc in np.arange(0.0001, 0.0011, 0.0001):    
    basic_result_tc = basic_result.copy()
    basic_result_tc.loc[basic_result_tc['market_sign'] == 1, 'Long'] -= (tc * 2)
    plt.plot(pd.to_datetime(basic_result.Trddt).dt.date, (1 + basic_result_tc['Long']).cumprod(), label = 'tc={}%'.format(round(tc * 100, 2)))
plt.plot(pd.to_datetime(basic_result.Trddt).dt.date, (1 + basic_result['Retindex']).cumprod(), '--',label = '沪深300', c = 'black')
plt.ylabel("累计净值")
plt.legend()
#plt.savefig("图2基础策略做多组合交易性成本分析.jpg",dpi = 1000)
plt.show()

plt.figure(figsize = (16, 8))
for tc in np.arange(0.0001, 0.0011, 0.0001):    
    basic_result_tc = basic_result.copy()
    basic_result_tc.loc[basic_result_tc['market_sign'] == 1, 'Long_Short'] -= (tc * 4)
    plt.plot(pd.to_datetime(basic_result.Trddt).dt.date, (1 + basic_result_tc['Long_Short']).cumprod(), label = 'tc={}%'.format(round(tc * 100, 2)))
plt.plot(pd.to_datetime(basic_result.Trddt).dt.date, (1 + basic_result['Retindex']).cumprod(), '--',label = '沪深300', c = 'red')
plt.ylabel("累计净值")
plt.legend()
#plt.savefig("图2基础策略多空组合交易性成本分析.jpg",dpi = 1000)
plt.show()

## 交易成本0.03%
basic_result_tc = basic_result.copy()
basic_result_tc.loc[basic_result_tc['market_sign'] == 0, 'Long'] -= (0.0003 * 2)
evaluate(basic_result_tc, 'Long', 'market_sign', basic_result_tc['Retindex'], 0.03, 252)

basic_result_tc = basic_result.copy()
basic_result_tc.loc[basic_result_tc['market_sign'] == 0, 'Long_Short'] -= (0.0003 * 4)
evaluate(basic_result_tc, 'Long_Short', 'market_sign', basic_result_tc['Retindex'], 0.03, 252)

## 剔除流动性策略收益率序列数据
lack_liquility = pd.read_csv("./无交易成本买卖前10后10只股票无流动性择时.csv")

## 剔除流动性策略评估
plt.figure(figsize = (16, 8))
plt.plot(pd.to_datetime(lack_liquility.Trddt).dt.date, (1+lack_liquility['Long']).cumprod(), label = '做多')
plt.plot(pd.to_datetime(lack_liquility.Trddt).dt.date, (1+lack_liquility['Long_Short'] ).cumprod(), label = '多空组合')
plt.plot(pd.to_datetime(lack_liquility.Trddt).dt.date, (1 + lack_liquility['Retindex']).cumprod(), label = '沪深300')
plt.legend()
plt.ylabel('累计净值')
# plt.savefig("图3无流动性择时累计净值.jpg", dpi = 1000)

evaluate(lack_liquility, 'Long', '', lack_liquility['Retindex'], 0.03, 252)
evaluate(lack_liquility, 'Long_Short', '', lack_liquility['Retindex'], 0.03, 252)

## 剔除流动性策略交易成本分析
plt.figure(figsize = (16, 8))
for tc in np.arange(0.0001, 0.0011, 0.0001):    
    lack_liquility_tc = lack_liquility.copy()
    lack_liquility_tc.loc[lack_liquility_tc['market_sign'] == 1, 'Long'] -= (tc * 2)
    plt.plot(pd.to_datetime(lack_liquility_tc.Trddt).dt.date, (1 + lack_liquility_tc['Long']).cumprod(), label = 'tc={}%'.format(round(tc * 100, 2)))
plt.plot(pd.to_datetime(lack_liquility.Trddt).dt.date, (1 + lack_liquility['Retindex']).cumprod(), '--',label = '沪深300', c = 'black')
plt.ylabel("累计净值")
plt.legend()
# plt.savefig("图4无流动性择时做多组合交易性成本分析.jpg",dpi = 1000)
plt.show()

plt.figure(figsize = (16, 8))
for tc in np.arange(0.0001, 0.0011, 0.0001):    
    lack_liquility_tc = lack_liquility.copy()
    lack_liquility_tc.loc[lack_liquility_tc['market_sign'] == 1, 'Long_Short'] -= (tc * 4)
    plt.plot(pd.to_datetime(lack_liquility_tc.Trddt).dt.date, (1 + lack_liquility_tc['Long_Short']).cumprod(), label = 'tc={}%'.format(round(tc * 100, 2)))
plt.plot(pd.to_datetime(lack_liquility.Trddt).dt.date, (1 + lack_liquility['Retindex']).cumprod(), '--',label = '沪深300', c = 'black')
plt.ylabel("累计净值")
plt.legend()
# plt.savefig("图4无流动性择时多空组合交易性成本分析.jpg",dpi = 1000)
plt.show()

lack_liquility_tc = lack_liquility.copy()
lack_liquility_tc.loc[lack_liquility_tc['market_sign'] == 1, 'Long'] -= (0.0003 * 2)
evaluate(lack_liquility_tc, 'Long', 'market_sign', lack_liquility_tc['Retindex'], 0.03, 252)

lack_liquility_tc = lack_liquility.copy()
lack_liquility_tc.loc[lack_liquility_tc['market_sign'] == 1, 'Long_Short'] -= (0.0003 * 4)
evaluate(lack_liquility_tc, 'Long_Short', 'market_sign', lack_liquility_tc['Retindex'], 0.03, 252)