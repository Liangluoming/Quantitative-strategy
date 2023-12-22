"""
@author: Liang Luoming
@email: liangluoming00@163.com
"""

"""
方正证券研报《A 股“跳一跳”:隔夜跳空选股因子》复现

数据来源：
    个股交易数据：CSMAR
    沪深300指数：CSMAR
    沪深300指数成分股数据：CSMAR
    中证500指数成分股数据：锐思
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import datetime as dt 
import warnings 
from scipy import stats
from tqdm import tqdm
warnings.filterwarnings('ignore')
import os
os.chdir("～")

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

def rolling_mean(data, col, N):
    """
    计算窗口均值
    Args:
        N: 窗口大小
    """
    return data[col].rolling(window = N).mean()

def Factor_Jump(data, N, opnname = 'Opnprc', prename = 'PreClosePrice'):
    """
    隔夜跳空因子构建
    Args:
        opnname: 当天交易日开盘价特征名
        prename: 前一交易日收盘价特征名
    """
    df = data.copy()
    df['ft'] = df[opnname] / (df[prename]) - 1 ## 计算隔夜涨跌幅
    df['|ft|'] = np.abs(df['ft']) ## 隔夜涨跌幅的绝对值
    df['Momentum_Ovn_N'] = df.groupby('Stkcd').apply(rolling_mean, 'ft', N).droplevel(0) ## 计算过去N个交易日隔夜涨跌幅的平均值
    df['Jump_N'] = df.groupby('Stkcd').apply(rolling_mean, '|ft|', N).droplevel(0)  ## 计算过去N个交易日的隔夜跳空因子
    return df['Momentum_Ovn_N'], df['Jump_N']


def get_MonthData(data, tname, type = 'last'):
    """
    从日度数据中获取月初与月末交易日数据
    Args:
        tname: 时间变量
        type: "first": 月初; "last": 月末
    """
    df = data.copy()
    df.sort_values(by = tname, ignore_index = True, inplace = True)
    timetable = pd.DataFrame({tname : data[tname].unique()})
    timetable['Year'] = timetable[tname].apply(lambda x : x.year)
    timetable['Month'] = timetable[tname].apply(lambda x : x.month)

    ## 根据type获取一个月初（月末）交易日的时间表
    timetable =  timetable.drop_duplicates(subset = ['Year', 'Month'], keep = type).sort_values(by = 'Trddt', ignore_index = True) 

    ## 从日度数据中获取在时间表上的数据即为月初（月末）数据
    df = data[data[tname].isin(timetable[tname])]
    return df

def get_ChangeRatio(data, col):
   """
   计算上一交易日买入当次交易日卖出的收益率
   """

   return data[col] / data[col].shift(1) - 1

"""
====================================================================================================================================
                                                           买卖策略
- 月末根据过去N天的因子水平进行10分组排序
- 月初在收盘时以收盘价按分组买入股票，同时以收盘价卖出股票池中的股票（避免涨停、停牌不买入，跌停、停牌不卖出）
- 收益率计算：（卖出价格 / 买入价格 - 1）

分层回溯：研报测算了10分组每一组对下一个月的收益率，但这存在一个问题：
    - 股票池中的股票由于涨停、跌停、停牌原因无法卖出，在后续可以卖出的时候其收益率计算为当期收益率还是后续卖出时的收益率？
    - 第一次尝试：将未卖出的股票保留在股票池中，并调整新买入股票的权重，在后续卖出未及时卖出的股票时，其收益率算卖出时的收益率，复现结果与研报不符。
    - 第二次尝试：不计算为卖出股票的收益率，不调整新买入股票的权重，即忽略掉未卖出这部分的股票（视为在后续可以卖的时候及时卖出），复现结果与研报相似。
    - 因此复现采用第二次尝试的买卖策略。
====================================================================================================================================

"""

def select_stocks(data, sortname, weighted = '', groups_num = 10):
    """
    月末根据因子表现确定股票分组
    Args:
        sortname: 因子特征名
        weighted: ''表示等权重; 其他表示按某一特征进行加权
    """
    df = data.copy()
    df['Group'] = pd.qcut(df[sortname], q = groups_num, labels = False) ## 分组
    ## 计算权重
    if weighted == '':
        df['Weight'] = 1 / len(df)
    else:
        df['Weight'] = df[weighted] / df.groupby(['Group'])[weighted].transform('sum')

    df.sort_values(by = ['Group', 'Stkcd'], ignore_index = True, inplace = True)

    return df[['Group', 'Stkcd', 'Weight']] ## 返回买入股票的分组

def execute_trades(data, pools, selected_stocks, sortname, cost = 0):
    """
    执行交易
    Args:
        data: 月初交易日数据，包含是否停牌、涨跌停、收盘价
        pools: 股票池（考虑未卖出股票时候需要用到）
        selected_stocks: 根据月末因子表现排好序的股票
        cost: 交易成本
    Returns:
        ret: 分组收益率
        pools: 股票池，包含了股票代码与持有权重
        sell_availiable: 卖出的股票
        buy_availiable: 买入的股票
        factor_group: 分组因子均值
    """
    df = data.copy()
    df = df[df['Stop'] == 0][df['Rangestop'] == 0] ## 去掉停牌、涨跌停的数据

    ## 先卖出后买入
    if len(pools):
        ## 股票池中存在股票才能够卖出
        sell_availiable = pd.merge(pools, df, on = ['Stkcd'])  ## 根据月初交易日数据确定能卖的股票
        pools = pd.DataFrame() ## 不考虑未卖出股票
        # pools = pools[ ~ pools['Stkcd'].isin(sell_availiable)]  ## 考虑未卖出的股票
        # sell_availiable['strategy_ret'] = ((sell_availiable['Adjprcwd'] / sell_availiable['Buyprc']) - 1 - cost) * sell_availiable['Weight']
        sell_availiable['strategy_ret'] = ((sell_availiable['ret'] - cost) * sell_availiable['Weight']) ## 计算收益率

        ret = sell_availiable.groupby('Group').sum().reset_index()  ## 计算分组收益率

        ret = ret[['Group', 'strategy_ret']]

        sell_availiable = sell_availiable[['Group', 'Stkcd', 'Weight', 'Adjprcwd', 'Buyprc']]
    else:
        ## 假如股票池中不存在股票那么收益率设为0
        groups = selected_stocks['Group'].unique()
        ret = pd.DataFrame({'Group': groups, 'strategy_ret' : [0] * len(groups)})
        sell_availiable = pd.DataFrame()

    buy_availiable = pd.merge(selected_stocks, df, on = ['Stkcd']) ## 根据月初交易日数据确定能买的股票
    buy_availiable.rename(columns = {'Adjprcwd' : 'Buyprc'}, inplace = True)

    ## 考虑未卖出的股票时候，需要根据股票池中的股票进行权重调整
    # if len(pools):
    #     pools_weight = pools.groupby('Group').sum().reset_index()
    #     pools_weight['availiable_weight'] = 1 - pools_weight['Weight']
    #     buy_availiable = pd.merge(buy_availiable, pools_weight[['Group', 'availiable_weight']], on = 'Group', how = 'left')
    #     buy_availiable['availiable_weight'].fillna(1, inplace = True)
    #     buy_availiable['Weight'] = buy_availiable['Weight'] * buy_availiable['availiable_weight']

    ## 往股票池中添加买入的股票
    pools = pd.concat([pools, buy_availiable[['Group', 'Stkcd', 'Weight', 'Buyprc']]])

    ## 计算分组因子的平均值
    factor_group = buy_availiable.groupby('Group').mean().reset_index()[['Group', sortname]]

    return ret, pools, sell_availiable, buy_availiable, factor_group


def strategy(data_start, data_end, sortname, time_name, weighted = '', groups_num = 10, cost = 0):
    """
    策略:
    Args:
        data_start: 月初交易日数据
        data_end: 月末交易日数据
    """
    df_end = data_end.copy()
    df_start = data_start.copy()
    df_end.sort_values(by = time_name, ignore_index = True, inplace = True)
    df_start.sort_values(by = time_name, ignore_index = True, inplace = True)
    pools = pd.DataFrame()
    rets, sells, buys, factors = [], [], [], []
    for t_end, t_start in tqdm(zip(df_end[time_name].unique(), df_start[time_name].unique())):
        df_end_t = df_end[df_end[time_name] == t_end].reset_index(drop = True) ## 月末数据
        df_start_t = df_start[df_start[time_name] == t_start].reset_index(drop = True) ## 月初数据
        selected_stocks = select_stocks(df_end_t, sortname, weighted, groups_num) ## 根据月末数据确认月初考虑分组购买的股票
        ret, pools, sell, buy, factor = execute_trades(df_start_t, pools, selected_stocks, sortname, cost) ## 执行交易
        ret[time_name], sell[time_name], buy[time_name], factor[time_name] = t_start, t_start, t_start, t_start ## 记录交易时间
        ## 储存 收益、买卖记录、因子值
        rets.append(ret)
        sells.append(sell)
        buys.append(buy)
        factors.append(factor)
    return rets, sells, buys, factors



def calculate_drawdown(series):
    """
    计算回撤
    """
    max_value = series[0]
    drawdown = np.zeros(len(series))

    for i in range(1, len(series)):
        if series[i] > max_value:
            max_value = series[i]
        drawdown[i] = (max_value - series[i]) / max_value

    return drawdown

def strategy_GroupTest(rets, path = ''):
    """
    分组回溯
    Args:
        path: 用于储存图片的路径
    """
    year_rets = (1 + rets).prod().apply(lambda x : x ** (12 / len(rets))) - 1 ## 计算年化收益率

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4)) ## 绘制分组回溯的图
    ## 分组年化收益率的柱状图
    ax1.bar(x = np.arange(1, 11), height = year_rets, width = 0.8, edgecolor = 'black')
    ax1.set_xticks(np.arange(1,11)) ## 设置x轴的刻度
    ax1_twin = ax1.twinx() ## 双y轴
    winrate = (rets>0).sum(axis = 0) / len(rets) ##计算胜率
    ax1_twin.plot(np.arange(1, 11), winrate, color = 'red') ## 绘制胜率

    ## 多空组
    longshort_ret = 0.5 * (rets[9] - rets[0])
    maxDrawdown = calculate_drawdown(longshort_ret) ## 计算多空组合的回撤
    ax2.plot(rets.index, (0.5 * (rets[9] - rets[0]) + 1).cumprod(), color = 'red') ## 多空组的累计收益
    ax2_twin = ax2.twinx() ## 双y轴
    ax2_twin.fill_between(rets.index, 0, - maxDrawdown, color='grey', alpha=0.7, label='Drawdown Area') ## 绘制多空组合的回撤
    plt.tight_layout()
    ## 保存图片
    if path != '':
        plt.savefig(path, dpi = 1000)
    plt.show()

# 计算因子相关系数
def get_factor_corr(df, factorList, RetName, method='spearman'):
    """
    计算因子相关系数
    """
    corr_Res = df[factorList].corrwith(df[RetName], method=method)

    return corr_Res

def cal_factor_ic(df, TimeName, factorList, method='spearman'):
    """
    计算ic序列
    """
    ic = df.groupby(TimeName).apply(get_factor_corr, factorList, 'ret', method)

    return ic

def get_max_drawdownRate(ret):
    # 最大回撤
    ret = (ret + 1).cumprod()
    endDate = np.argmax((np.maximum.accumulate(ret)) - ret)
    if endDate == 0:
        return 0, len(ret), endDate
    else:
        startDate = np.argmax(ret[:endDate])
    return (ret[startDate] - ret[endDate]) / ret[startDate], startDate, endDate

def get_annual_ret(ret, freq):
    # 年化收益率
    annual_ret = (ret + 1).prod() ** (freq / len(ret)) - 1
    return annual_ret


def get_annual_vol(ret, freq):
    # 年化波动率
    annual_vol = np.sqrt(freq) * ret.std()
    return annual_vol

def get_win_rate(ret):
    """
    计算胜率
    """
    win_rate = np.sum(ret>0) / len(ret)
    
    return win_rate


# 计算Beta、Alpha
def get_alpha_beta(df, port_col, bench_col, rf, m):
    """
    计算 Beta、 Alpha
    """
    beta = df[port_col].cov(df[bench_col]) / df[bench_col].var()
    alpha = (get_annual_ret(df[port_col], m) - rf) - beta * (get_annual_ret(df[bench_col], m) - rf)

    return alpha, beta


def evaluate(month_start, month_end, factorList):
    """
    指标回测
    Args:
        month_start: 月初交易数据
        month_end: 月末交易数据
        factorList: 需要回测的因子列表
    Returns:
        GroupTestRets: 分组的年化收益率
        performance: 不同因子的多组在不同指标下的表现
    """
    Rets = []  ## 储存收益
    Factors = [] ## 储存因子名字
    ## 储存IC值、秩IC值、年化ICIR、年化收益率、年化波动率、IR、胜率、最大回撤率
    ICs, RankICs, ICIR, AnnualRet, AnnualVol, IR, WinRate, MaxDrawDown = [],[],[],[],[],[],[],[]

    ## 获取月份数据，用于后面计算ic时候将数据拼起来
    month_start['Year'] = month_start['Trddt'].apply(lambda x : x.year)
    month_start['Month'] = month_start['Trddt'].apply(lambda x : x.month)
    month_end_shift = month_end.copy()
    month_end_shift['Trddt'] = month_end_shift['Trddt'] + dt.timedelta(days = 28)
    month_end_shift['Year'] = month_end_shift['Trddt'].apply(lambda x : x.year)
    month_end_shift['Month'] = month_end_shift['Trddt'].apply(lambda x : x.month)

    for factor_name in factorList:
        Factors.append(factor_name)

        ## 分组回测
        rets, sells, buys, factors = strategy(month_start[month_start[factor_name].notnull()], month_end[month_end[factor_name].notnull()], factor_name, 'Trddt', 'Dsmvosd')
        rets_pivot = pd.concat(rets).pivot(index = 'Trddt', columns = 'Group', values = 'strategy_ret')
        path = "{}分层回溯.jpg".format(factor_name)
        strategy_GroupTest(rets_pivot, path)

        ## 多空组合
        rets_pivot['longshort'] = 0.5 * (rets_pivot[9] - rets_pivot[0])
        rets_npv = ((rets_pivot + 1).prod()).apply(lambda x : x ** (12 / len(rets_pivot)) - 1)
        Rets.append((pd.DataFrame(rets_npv).T).rename(index = {0 : factor_name}))


        ## 计算IC……
        ic_df = pd.merge(month_start[['Stkcd', 'Trddt','Year', 'Month', 'ret']], month_end_shift[['Stkcd', 'Year', 'Month', factor_name]])

        ic_series = cal_factor_ic(ic_df[ic_df[factor_name].notnull()], 'Trddt', [factor_name], 'pearson')
        ic_series_rank = cal_factor_ic(ic_df[ic_df[factor_name].notnull()], 'Trddt', [factor_name], 'spearman')
        ic_mean = np.nanmean(ic_series)
        ic_std = np.nanstd(ic_series)
        ic_rank_mean = np.nanmean(ic_series_rank)
        icir = ic_mean / ic_std * np.sqrt(12)
        buys = pd.concat(buys)
        buys = buys[buys['Group'] == 0]
        buys = buys.groupby('Trddt').count().reset_index()
        annual_ret = get_annual_ret(rets_pivot[9], 12)
        annual_vol = get_annual_vol(rets_pivot[9], 12)
        annual_ir = np.mean(rets_pivot[9]) / np.std(rets_pivot[9]) * np.sqrt(12)
        win_rate = get_win_rate(rets_pivot[9])
        maxdrawdownrate, _, __  = get_max_drawdownRate(rets_pivot[9])
        ICs.append(ic_mean)
        RankICs.append(ic_rank_mean)
        ICIR.append(icir)
        AnnualRet.append(annual_ret)
        AnnualVol.append(annual_vol)
        IR.append(annual_ir)
        WinRate.append(win_rate)
        MaxDrawDown.append(maxdrawdownrate)

    GroupTestRets = pd.concat(Rets)
    performance = pd.DataFrame({'因子名称': Factors,
                'IC均值': ICs,
                '秩IC均值': RankICs,
                '年化ICIR': ICIR,
                '年化收益': AnnualRet,
                '年化波动': AnnualVol,
                '年化IR': IR,
                '胜率': WinRate,
                '最大回撤': MaxDrawDown})

    return GroupTestRets, performance


paths = get_paths('./个股交易数据/', '.csv') ## 获取保存个股交易数据的路径

datas = [pd.read_csv(path) for path in paths]  ## 读入个股交易数据
raw_data = pd.concat(datas)
raw_data['Stkcd'] = raw_data['Stkcd'].apply(lambda x : str(x).zfill(6))
raw_data = raw_data[raw_data['Markettype'].isin([1, 4, 16, 32, 64])] ## 全部A股

## 读入公司上市日期数据
list_df = pd.read_csv("./TRD_Co.csv")
list_df['Stkcd'] = list_df['Stkcd'].apply(lambda x : str(x).zfill(6))

raw_data = pd.merge(raw_data, list_df[['Stkcd', 'Listdt']], on = ['Stkcd'], how = 'left')

## 转成date时间数据格式
raw_data['Trddt'] = pd.to_datetime(raw_data['Trddt']).dt.date
raw_data['Listdt'] = pd.to_datetime(raw_data['Listdt']).dt.date

# 停牌(CSMAR的个股数据不包含停牌数据)
stop = pd.read_csv("./TSR_Stkstat.csv")
stop['Stkcd'] = stop['Stkcd'].apply(lambda x : str(x).zfill(6))
stop['Resmdate'] = pd.to_datetime(stop['Resmdate']).dt.date
stop['Suspdate'] = pd.to_datetime(stop['Suspdate']).dt.date

## 计算停牌至复牌的间隔时间
stop['Resdtdiff'] = (stop['Resmdate'] - stop['Suspdate']).apply(lambda x : x.days)
stop = stop[['Stkcd', 'Suspdate', 'Resdtdiff']]

## Stop取值为1表示停牌，取值为0表示不停牌
stop['Stop'] = 1
raw_data = pd.merge(raw_data, stop, left_on = ['Stkcd', 'Trddt'], right_on = ['Stkcd', 'Suspdate'], how = 'left')
raw_data['Stop'].fillna(0, inplace = True)

## 根据最低价和最高价价去判断是否涨跌停（由于数据的原因，用正负10%去判断存在一定的问题，因为一些涨停的收益率增幅不足10%）
raw_data['Rangestop'] = (raw_data['Hiprc'] == raw_data['Loprc']).astype('int')
# data.loc[data[(data['ChangeRatio']<=0.0999) & (data['ChangeRatio']>= -0.0999)].index, 'Rangestop'] = 0
# data['Rangestop'].fillna(1, inplace = True)

## 计算交易时间距离上市时间的间隔天数
raw_data['Listeddt'] = (raw_data['Trddt'] - raw_data['Listdt']).apply(lambda x : x.days) 
raw_data = raw_data[raw_data['Listeddt'] >= 49] ## 剔除上市未满50日的新股

## 剔除当日交易额小于500万的股票
raw_data = raw_data[raw_data['Dnvaltrd'] >= 5000000]
## 计算换手率
raw_data['turnover'] = raw_data['Dnshrtrd'] / (raw_data['Dsmvosd'] * 1000 / raw_data['Clsprc'])
## 剔除当日换手率不足0.1%的股票
raw_data = raw_data[raw_data['turnover'] >= 0.001]

"""
注意CSMAR的数据存在一些问题，有一些股票中间缺失很长一段时间。在初始阶段我尝试的是将这些数据去除，但实践后发现结果和研报复现存在gap，因此最终版本不再除去这些数据，但在后续会有调整。
"""

# def getdtdiff(data, col):
#     return (data[col].diff()).apply(lambda x : x.days)

# raw_data = raw_data[~raw_data['Stkcd'].isin(raw_data['Stkcd'].value_counts()[raw_data['Stkcd'].value_counts()<5].index)]
# raw_data.sort_values(by = ['Trddt', 'Stkcd'], ignore_index = True, inplace = True)
# raw_data['dtdiff'] = raw_data.groupby('Stkcd').apply(getdtdiff, 'Trddt').droplevel(0)

# raw_data.sort_values(by = ['Stkcd', 'Trddt'], ignore_index= True, inplace = True)
# unnormal_index = raw_data[raw_data['dtdiff'] > 160].drop_duplicates(subset = ['Stkcd'], keep = 'last').index.to_numpy()
# unnormal_stkcd = raw_data[raw_data['dtdiff'] > 160].drop_duplicates(subset = ['Stkcd'], keep = 'last')['Stkcd'].values
# unnormal_index_start = raw_data[raw_data['Stkcd'].isin(unnormal_stkcd)].drop_duplicates(subset = 'Stkcd', keep = 'first').index.to_numpy()
# for i in tqdm(range(len(unnormal_index))):
#     raw_data.drop(index = np.arange(unnormal_index_start[i],unnormal_index[i]), inplace = True)

## 选择回测时间。（回测时间应为2006年1月1日开始，为了计算因子值，将时间保留至2005年1月1日）
data = raw_data[(raw_data['Trddt']>=dt.date(2005, 1, 1)) & (raw_data['Trddt']<=dt.date(2018, 6, 30))]


## 计算因子
data.sort_values(by = ['Trddt', 'Stkcd'], ignore_index = True, inplace = True)
data['Momentum_Ovn_10'], data['Jump_10'] = Factor_Jump(data, 10)
data['Momentum_Ovn_60'], data['Jump_60'] = Factor_Jump(data, 60)
data['Momentum_Ovn_120'], data['Jump_120'] = Factor_Jump(data, 120)

"""
==============
   描述性分析
==============

"""

"""

            图表4

"""

df = data.copy()
df = df[df['Trddt'] == dt.date(2018, 6, 29)]
plt.hist(df['Momentum_Ovn_10'],bins = 51, edgecolor = 'white', facecolor = 'red')
#plt.savefig("图表4:2018年6月29日隔夜涨跌幅水平统计结果", dpi = 1000)
plt.show()

print("数据量:{}".format(len(df[df['Momentum_Ovn_10'].notnull()]['Momentum_Ovn_10'])))
print("均值: %.4f" % (df['Momentum_Ovn_10'].mean()))
print("标准差: %.4f" % (df['Momentum_Ovn_10'].std()))
print("中位数: %.4f" % (df['Momentum_Ovn_10'].median()))
print("偏度: %.4f" % (stats.skew(df['Momentum_Ovn_10'], nan_policy = 'omit')))
print("峰度: %.4f" % (stats.kurtosis(df['Momentum_Ovn_10'], nan_policy = 'omit')))


"""

            图表5
            
"""

df = get_MonthData(data, 'Trddt', type = 'last')
df = df[(df['Trddt'] >= dt.date(2006, 1, 1)) & (df['Trddt'] <= dt.date(2018, 6, 30))]
plt.hist(df['Momentum_Ovn_10'], bins = 600, edgecolor = 'white', facecolor = 'red')
#plt.savefig("图表5:2016年初-2018年中隔夜涨跌幅水平统计结果.jpg", dpi = 1000)
plt.xlim(-0.1, 0.1)
plt.show()

print("数据量:{}".format(len(df[df['Momentum_Ovn_10'].notnull()]['Momentum_Ovn_10'])))
print("均值: %.4f" % (df['Momentum_Ovn_10'].mean()))
print("标准差: %.4f" % (df['Momentum_Ovn_10'].std()))
print("中位数: %.4f" % (df['Momentum_Ovn_10'].median()))
print("偏度: %.4f" % (stats.skew(df['Momentum_Ovn_10'], nan_policy = 'omit')))
print("峰度: %.4f" % (stats.kurtosis(df['Momentum_Ovn_10'], nan_policy = 'omit')))

"""

            图表7
            
"""

df = data.copy()
df = df[df['Trddt'] == dt.date(2018, 6, 29)]
plt.hist(df['Jump_10'],bins = 46, edgecolor = 'white', facecolor = 'red')
#plt.savefig("图表7:2018年6月29日跳空因子统计结果.jpg", dpi = 1000)
plt.xlim(0, 0.08)
plt.ylim(0, 600)
plt.show()

print("数据量:{}".format(len(df[df['Jump_10'].notnull()]['Jump_10'])))
print("均值: %.4f" % (df['Jump_10'].mean()))
print("标准差: %.4f" % (df['Jump_10'].std()))
print("中位数: %.4f" % (df['Jump_10'].median()))
print("偏度: %.4f" % (stats.skew(df['Jump_10'], nan_policy = 'omit')))
print("峰度: %.4f" % (stats.kurtosis(df['Jump_10'], nan_policy = 'omit')))

"""

            图表8
            
"""

## 根据研报：图表8用的是月末数据
df = get_MonthData(data, "Trddt", type = 'last') ## 获取月末交易数据
df = df[(df['Trddt']>=dt.date(2006, 1,1)) & (df['Trddt'] <= dt.date(2018, 6, 30))]
plt.hist(df['Jump_10'], bins = 1000, edgecolor = 'white', facecolor = 'red')
plt.xlim(0, 0.1)
plt.ylim(0, 45000)
#plt.savefig("图表8:2016年初-2018年中跳空因子统计结果.jpg", dpi = 1000)
plt.show()

print("数据量:{}".format(len(df[df['Jump_10'].notnull()]['Jump_10'])))
print("均值: %.4f" % (df['Jump_10'].mean()))
print("标准差: %.4f" % (df['Jump_10'].std()))
print("中位数: %.4f" % (df['Jump_10'].median()))
print("偏度: %.4f" % (stats.skew(df['Jump_10'], nan_policy = 'omit')))
print("峰度: %.4f" % (stats.kurtosis(df['Jump_10'], nan_policy = 'omit')))



"""

            因子分层回溯
            
"""


"""

            图表6
            
"""

## 获取月初和月末数据
month_start = get_MonthData(data, "Trddt", type = "first")
month_start = month_start[(month_start['Trddt']>=dt.date(2005, 12, 1)) & (month_start['Trddt']<=dt.date(2018, 6, 30))].reset_index(drop = True)

month_end = get_MonthData(data, "Trddt", type = "last")
month_end = month_end[(month_end['Trddt']>=dt.date(2005, 12, 1)) & (month_end['Trddt']<=dt.date(2018, 6, 30))].reset_index(drop = True)


## 计算收益率
month_start['ret'] = month_start.groupby('Stkcd').apply(get_ChangeRatio, 'Adjprcwd').droplevel(0)

## 根据前面所说，CSMAR数据存在部分股票的时间维度上不连续，导致计算出来的收益率存在超500%的，因此做调整
month_start = month_start[(month_start['ret']>= np.nanquantile(month_start['ret'], 1 - 0.999)) & (month_start['ret']<= np.nanquantile(month_start['ret'], 0.999))]

## 绘制图表6
Rets = []
for N in [10, 60, 120]:
    factor_name = 'Momentum_Ovn_{}'.format(N)
    rets, sells, buys, factors = strategy(month_start[month_start[factor_name].notnull()], month_end[month_end[factor_name].notnull()], factor_name, 'Trddt', 'Dsmvosd')
    rets_pivot = pd.concat(rets).pivot(index = 'Trddt', columns = 'Group', values = 'strategy_ret')
    rets_npv = (rets_pivot + 1).prod()
    Rets.append(rets_npv)

plt.figure(figsize = (12, 6))
columns = np.arange(10)
plt.bar(x = np.arange(1, 11) - 0.3, height = Rets[0], facecolor = 'red', width = 0.2, edgecolor = 'black')
plt.bar(x = np.arange(1, 11) - 0.05, height = Rets[1], facecolor = '#5A6133', width = 0.2, edgecolor = 'black')
plt.bar(x = np.arange(1, 11) + 0.2, height = Rets[2], facecolor = 'grey', width = 0.2, edgecolor = 'black')
plt.xticks(np.arange(1, 11))  
plt.legend(['Momentum_Ovn_10', 'Momentum_Ovn_60', 'Momentum_Ovn_120'])
#plt.savefig("图表6:隔夜涨跌幅水平分组收益.jpg", dpi = 1000)
plt.show()

"""

            图表9
            
"""

rets, sells, buys, factors = strategy(month_start[month_start['Jump_10'].notnull()], month_end[month_end['Jump_10'].notnull()], 'Jump_10', 'Trddt', 'Dsmvosd')
Factor_Jump10 = pd.concat(factors).groupby('Group').mean().reset_index()


fig, ax1 = plt.subplots()
plt.grid(axis = 'y')
ax1.plot(np.arange(1, 11), Rets[0].apply(lambda x : x ** (12 / 150) - 1), color = 'red')

ax1.set_ylim(0, 0.25)
ax1.set_yticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
ax1.set_yticklabels(['0%', '5%', '10%', '15%', '20%', '25%'])
ax2 = ax1.twinx()
ax2.bar(x = Factor_Jump10['Group'] + 1, height = Factor_Jump10['Jump_10'], width = 0.6, edgecolor = 'black')
ax2.set_xticks(np.arange(1, 11))
ax2.set_ylim(0, 0.02)
ax1.scatter(np.arange(1, 11), Rets[0].apply(lambda x : x ** (12 / 150) - 1), color = 'red')
#plt.savefig("图表9：10日跳空因子分组收益.jpg", dpi = 1000)
plt.show()

"""

            因子测试结果：多空净值、胜率、年化收益。图表11-23
            
"""


## 计算因子
data.sort_values(by = ['Trddt', 'Stkcd'], ignore_index = True, inplace = True)
for N in [5, 10, 20, 40, 60, 120]:
   _, data['Adj_Jump_{}'.format(N)] = Factor_Jump(data, N)
   data['Adj_Jump_{}'.format(N)] = - data['Adj_Jump_{}'.format(N)]

## 获取月初和月末数据
month_start = get_MonthData(data, "Trddt", type = "first")
month_start = month_start[(month_start['Trddt']>=dt.date(2005, 12, 1)) & (month_start['Trddt']<=dt.date(2018, 6, 30))].reset_index(drop = True)

month_end = get_MonthData(data, "Trddt", type = "last")
month_end = month_end[(month_end['Trddt']>=dt.date(2005, 12, 1)) & (month_end['Trddt']<=dt.date(2018, 6, 30))].reset_index(drop = True)

month_start['ret'] = month_start.groupby('Stkcd').apply(get_ChangeRatio, 'Adjprcwd').droplevel(0)

month_start = month_start[(month_start['ret']>= np.nanquantile(month_start['ret'], 1 - 0.999)) & (month_start['ret']<= np.nanquantile(month_start['ret'], 0.999))]


"""
    图表11-22
"""

factorList = ['Jump_{}'.format(N) for N in [5, 10, 20, 40, 60, 120]]
GroupTestRets, performance = evaluate(month_start, month_end, factorList)

"""
    图表23
"""
GroupTestRets

"""
    图表30
"""

performance



"""
==================
      策略拓展
==================
"""


"""
回测区间拓展到2023年，评估最近1年、3年的因子表现
"""

## 2018年中后的数据
test_data = raw_data[(raw_data['Trddt'] >= dt.date(2018, 6, 1))]
test_data.sort_values(by = ['Trddt', 'Stkcd'], ignore_index = True, inplace = True)

## 计算因子
for N in [5, 10, 20, 40, 60, 120]:
   _, test_data['Jump_{}'.format(N)] = Factor_Jump(data, N)
   test_data['Adj_Jump_{}'.format(N)] = - test_data['Jump_{}'.format(N)]


## 获取月初、月末交易数据
month_start = get_MonthData(test_data, "Trddt", type = "first")
month_start = month_start.reset_index(drop = True)

month_end = get_MonthData(test_data, "Trddt", type = "last")
month_end = month_end.reset_index(drop = True)

month_start['ret'] = month_start.groupby('Stkcd').apply(get_ChangeRatio, 'Adjprcwd').droplevel(0)

month_start = month_start[(month_start['ret']>= np.nanquantile(month_start['ret'], 1 - 0.999)) & (month_start['ret']<= np.nanquantile(month_start['ret'], 0.999))]


factorList = ['Jump_{}'.format(N) for N in [5, 10, 20, 40, 60, 120]]
GroupTestRets, performance = evaluate(month_start, month_end, factorList)

GroupTestRets

performance

"""
以沪深300指数为基准，评估隔夜跳空选股因子的表现
"""
## 读入沪深300指数数据
index300 = pd.read_csv("./TRD_Index.csv")
index300['Indexcd'] = index300['Indexcd'].apply(lambda x : str(x).zfill(6))
index300 = index300[index300['Indexcd'] == "000300"]  ## 选取沪深300
index300['Trddt'] = pd.to_datetime(index300['Trddt']).dt.date
index300.sort_values(by = 'Trddt', ignore_index = True, inplace = True)

index300_pivot = index300.pivot(index = 'Trddt', columns = 'Indexcd', values = 'Clsindex').shift(1).unstack().reset_index().rename(columns = {0 : 'Preindex'})
index300 = pd.merge(index300, index300_pivot, on = ['Indexcd', 'Trddt'], how = 'left')

## 计算因子数据
index300['ft'] = index300['Opnindex'] / index300['Preindex'] - 1
index300['|ft|'] = np.abs(index300['ft'])
for N in [5, 10, 20, 40, 60, 120]:
   index300['Jump_{}'.format(N)] = index300['|ft|'].rolling(window = N).mean()
   index300['Adj_Jump_{}'.format(N)] = - index300['Jump_{}'.format(N)]


## 获取沪深300月初和月末交易数据
index300_month_start = get_MonthData(index300, 'Trddt', 'first')
index300_month_end = get_MonthData(index300, 'Trddt', 'last')

index300_month_start['ret'] = index300_month_start.groupby('Indexcd').apply(get_ChangeRatio, 'Opnindex').T.unstack().droplevel(0)

"""

评估的时候我们用所有的历史数据

"""


## 计算raw_data的因子
raw_data.sort_values(by = ['Trddt', 'Stkcd'], ignore_index = True, inplace = True)
for N in [5, 10, 20, 40, 60, 120]:
   _, raw_data['Adj_Jump_{}'.format(N)] = Factor_Jump(raw_data, N)
   raw_data['Adj_Jump_{}'.format(N)] = - raw_data['Adj_Jump_{}'.format(N)]

## 获取月初、月末交易数据
month_start = get_MonthData(raw_data, "Trddt", type = "first")
month_start = month_start.reset_index(drop = True)

month_end = get_MonthData(raw_data, "Trddt", type = "last")
month_end = month_end.reset_index(drop = True)

month_start['ret'] = month_start.groupby('Stkcd').apply(get_ChangeRatio, 'Adjprcwd').droplevel(0)
month_start = month_start[month_start['Trddt'] >= dt.date(2005, 5, 1)]
month_end = month_end[month_end['Trddt'] >= dt.date(2005, 4, 1)]

month_start = month_start[(month_start['ret']>= np.nanquantile(month_start['ret'], 1 - 0.999)) & (month_start['ret']<= np.nanquantile(month_start['ret'], 0.999))]


## 以沪深300为基准进行比较
Rets = []
for N in [5, 10, 20, 40, 60, 120]:
    actor_name = 'Adj_Jump_{}'.format(N)
    raw_factor_name = 'Jump_{}'.format(N)
    rets, sells, buys, factors = strategy(month_start[month_start[factor_name].notnull()], month_end[month_end[factor_name].notnull()], factor_name, 'Trddt', 'Dsmvosd')
    rets_pivot = pd.concat(rets).pivot(index = 'Trddt', columns = 'Group', values = 'strategy_ret')
    Rets.append(rets_pivot)

for i, N in enumerate([5, 10, 20, 40, 60, 120]):
    raw_factor_name = 'Jump_{}'.format(N)
    plt.plot(Rets[i].index, (Rets[i][9] + 1).cumprod(), label = raw_factor_name)
plt.plot(index300_month_start['Trddt'], (1 + index300_month_start['ret']).cumprod(), label = '沪深300')
plt.savefig("沪深300指数为基准评估隔夜跳空选股因子表现.jpg",dpi = 1000)
plt.legend()
plt.show()

## 计算信息比率
por_bench = Rets[2][9].values - index300_month_start[index300_month_start['Trddt']>=dt.date(2005,8,1)]['ret'].values
annual_ret = get_annual_ret(por_bench, 12)
annual_vol = get_annual_vol(por_bench, 12)
IR =  annual_ret/ annual_vol
print("信息比率:%.2f" %(IR))


## 以沪深300为基准计算隔夜跳空选股因子的Alpha
por = pd.DataFrame()
por['por'] = Rets[2][9].values
por['bench'] = index300_month_start[index300_month_start['Trddt']>=dt.date(2005,8,1)]['ret'].values
rf = 0.03
alpha, beta = get_alpha_beta(por, 'por', 'bench', rf, 12)
print("beta:%.2f" % beta)
print("alpha:%.2f" % alpha)

"""
使用沪深 300 或中证 500 成分股，重复前面的分析:表 31-37
"""

paths = get_paths('./指数成分股变更数据', '.csv') ## 获取沪深300成分股变更数据路径
## 读入指数成分股变更数据（这部分数据不包含中证500的数据）

indexchanges = [pd.read_csv(path) for path in paths]
index_change = pd.concat(indexchanges)
index_change_300 = index_change[index_change['Indexcd'] == '000300'].rename(columns = {'Chgsmp01' : 'Trddt', 'Chgsmp02' : 'Stkcd'})
index_change_300 = index_change_300[['Stkcd', 'Trddt', 'Chgsmp04']]
index_change_300['Trddt'] = pd.to_datetime(index_change_300['Trddt']).dt.date

index300_stkcd = pd.merge(raw_data, index_change_300, on = ['Stkcd', 'Trddt'], how = 'left')

## 注意这部分数据只有变更的状态：Chgsmp02: 1表示新增，2表示剔除
## 根据变更数据获取成分股
def fillna(data, col):
    return data[col].fillna(method = 'ffill')
index300_stkcd.sort_values(by = ['Trddt', 'Stkcd'], ignore_index = True, inplace = True)
index300_stkcd['Chgsmp04'] = index300_stkcd.groupby('Stkcd').apply(fillna, 'Chgsmp04').droplevel(0)

## 选取成分股数据
index300_stkcd = index300_stkcd[index300_stkcd['Chgsmp04'] == 1]


## 获取沪深300成分股月初和月末交易数据
index300_stkcd_start = get_MonthData(index300_stkcd, "Trddt", type = "first")
index300_stkcd_end = get_MonthData(index300_stkcd, "Trddt", type = "last")

index300_stkcd_start['ret'] = index300_stkcd_start.groupby('Stkcd').apply(get_ChangeRatio, 'Adjprcwd').droplevel(0)
index300_stkcd_start = index300_stkcd_start[(index300_stkcd_start['Trddt']>=dt.date(2006, 1, 1)) & (index300_stkcd_start['Trddt']<=dt.date(2018, 6, 30))]
index300_stkcd_end = index300_stkcd_end[(index300_stkcd_end['Trddt']>=dt.date(2005, 12, 1)) & (index300_stkcd_end['Trddt']<=dt.date(2018, 6, 30))]

index300_stkcd_start = index300_stkcd_start[(index300_stkcd_start['ret']>= np.nanquantile(index300_stkcd_start['ret'], 1 - 0.999)) & (index300_stkcd_start['ret']<= np.nanquantile(index300_stkcd_start['ret'], 0.999))]

index300_stkcd_start = index300_stkcd_start.reset_index(drop = True)
index300_stkcd_end = index300_stkcd_end.reset_index(drop = True)

"""
    沪深300成分股因子测试和IC评价
"""

factorList = ['Jump_{}'.format(N) for N in [5, 10, 20, 40, 60, 120]]
GroupTestRets, performance = evaluate(index300_stkcd_start, index300_stkcd_end, factorList)

GroupTestRets

performance


## 读取中证500成分股数据
index500 = pd.read_csv("./中证500.csv")
index500['Stkcd'] = index500['Stkcd'].apply(lambda x : str(x).zfill(6))
index500 = index500.melt(id_vars = 'Stkcd', value_vars = ['BegDt', 'EndDt'], var_name = 'flag', value_name = 'Trddt')
index500 = index500[index500['Trddt'].notnull()]
index500['Trddt'] = pd.to_datetime(index500['Trddt']).dt.date
index500['flag'] = index500['flag'].map({'BegDt': 1, 'EndDt' : 2})
index500_stkcd = pd.merge(raw_data, index500, on = ['Stkcd', 'Trddt'], how = 'left')


index500_stkcd.sort_values(by = ['Trddt', 'Stkcd'], ignore_index = True, inplace = True)
index500_stkcd['flag'] = index500_stkcd.groupby('Stkcd').apply(fillna, 'flag').droplevel(0)

## 保留中证500成分股数据
index500_stkcd = index500_stkcd[index500_stkcd['flag'] == 1]


## 中证500成分股月初和月末交易数据
index500_stkcd_start = get_MonthData(index500_stkcd, "Trddt", type = "first")

index500_stkcd_end = get_MonthData(index500_stkcd, "Trddt", type = "last")

index500_stkcd_start['ret'] = index500_stkcd_start.groupby('Stkcd').apply(get_ChangeRatio, 'Adjprcwd').droplevel(0)
index500_stkcd_start = index500_stkcd_start[(index500_stkcd_start['Trddt']>=dt.date(2007, 2, 1)) & (index500_stkcd_start['Trddt']<=dt.date(2018, 6, 30))]
index500_stkcd_end = index500_stkcd_end[(index500_stkcd_end['Trddt']>=dt.date(2007, 1, 1)) & (index500_stkcd_end['Trddt']<=dt.date(2018, 6, 30))]

index500_stkcd_start = index500_stkcd_start[(index500_stkcd_start['ret']>= np.nanquantile(index500_stkcd_start['ret'], 1 - 0.999)) & (index500_stkcd_start['ret']<= np.nanquantile(index500_stkcd_start['ret'], 0.999))]

index500_stkcd_start = index500_stkcd_start.reset_index(drop = True)
index500_stkcd_end = index500_stkcd_end.reset_index(drop = True)

"""
    中证500成分股因子测试和IC评价
"""

factorList = ['Jump_{}'.format(N) for N in [5, 10, 20, 40, 60, 120]]
GroupTestRets, performance = evaluate(index500_stkcd_start, index500_stkcd_end, factorList)

GroupTestRets

performance

