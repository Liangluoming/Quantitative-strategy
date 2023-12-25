"""
@author: LiangLuoming
@email: liangluoming00@163.com
@github: https://github.com/Liangluoming
"""

import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
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
def z_score(data, column):
    """
    标准化
    Args:
        data: 数据
        column: 标准化的特征
    Return:
        标准化后的数据
    """
    mi, ma = data[column].min(axis = 0), data[column].max(axis = 0)
    return (data[column] - mi) / (ma - mi) + 1

paths = get_paths("./raw_data/日度个股数据", '.csv') ## 读入日度个股数据
datas = [pd.read_csv(path) for path in paths] 
data = pd.concat(datas)
data = data[data['Markettype'].isin([1, 4])] ## 选取沪深A股数据，不包含科创板和创新版
data = data[data['Trdsta'] == 1] ## 正常交易状态数据

data['Stkcd'] = data['Stkcd'].apply(lambda x : str(x).zfill(6)) ## 补全证券代码
data.reset_index(drop = True, inplace = True)
data['Hi_Lo'] = (data['Hiprc'] - data['Loprc']) ## 计算最高最低价差

data['Hi_Lo'] = data.groupby('Trddt').apply(z_score, 'Hi_Lo').droplevel(0) ##最高最低价差标准化
data['Dnshrtrd_z'] = data.groupby('Trddt').apply(z_score, 'Dnshrtrd').droplevel(0) ##成交量标准化
data['D_D'] = data['Dnvaltrd'] / data['Dnshrtrd'] ## 流动性因子🪦
data['D_D'] = data.groupby('Trddt').apply(z_score, 'D_D').droplevel(0) ##流动性因子分母标准化
data['liquidity'] = (data['Dnshrtrd_z'] / data['Hi_Lo']) / data['D_D'] ## 计算流动性因子
data.drop(columns = ['Hi_Lo', 'Dnshrtrd_z', 'D_D'], inplace = True)

data.sort_values(by = ['Trddt', 'Stkcd'], ascending = [True, True], inplace = True, ignore_index=True)
data_pivot = data.pivot(index = 'Trddt', columns = 'Stkcd', values = 'Dretwd')
data_pivot = data_pivot.shift(-1).unstack().reset_index().rename(columns = {0 : 'F_Dretwd'})
data = pd.merge(data, data_pivot, on =['Trddt', 'Stkcd'], how = 'left')

data_pivot = data.pivot(index = 'Trddt', columns = 'Stkcd', values = 'liquidity')
data_pivot = data_pivot.shift(-1).unstack().reset_index().rename(columns = {0 : 'F_liquidity'})
data = pd.merge(data, data_pivot, on =['Trddt', 'Stkcd'], how = 'left')

indices = data.drop_duplicates(subset = ['Trddt'], keep = 'last').index.to_numpy() + 1
indices = np.append([0],indices)

dtime = data['Trddt'].unique()
path = "./股票时序交易数据/"
for i in tqdm(range(1, len(indices))):
    data.iloc[indices[i-1] : indices[i]].to_csv(path+"{}.csv".format(dtime[i - 1]), index = False)




# data.sort_values(by = ['Stkcd', 'Trddt'], ascending = [True, True], inplace = True, ignore_index=True)
# indices = data.drop_duplicates(subset = ['Stkcd'], keep = 'last').index.to_numpy() + 1
# indices = np.append([0],indices)

# codes = data['Stkcd'].unique()
# path = "/Users/luomingliang/Desktop/梁洛铭/研究生/UIBE/研一/Python与金融量化/期末大作业/数据/"
# for i in tqdm(range(1, len(indices))):
#     data.iloc[indices[i-1] : indices[i]].to_csv(path+"{}.csv".format(codes[i - 1]), index = False)