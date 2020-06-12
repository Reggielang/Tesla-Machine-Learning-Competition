# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:54:12 2020

@author: REGGIE
"""
#%% 导入数据
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\REGGIE\Desktop\Machine Learning Model Group')

df = pd.read_csv('site.csv')

#%% 数据的简单观察
print(df.info(),df.describe())

#%% 查看缺失值
df[df.var_value.isna()]
#10510/172191 缺失值只占了总数据的6%,使用均值填补
df.var_value = df.var_value.fillna(np.mean(df.var_value))

#%% 查看异常值
#a = df[df.var_value>1000]
#df = df.drop(df[df.var_value>1000].index)
#%%
#b = df[df.var_value<1]
#df = df.drop(df[df.var_value<1].index)
#%% 数据分布可视化--3个不同的充电站的数量可视化，a占据了绝大一部分，b占据了一小部分，c的数量几乎可以忽略
df.site_type.value_counts().plot(kind= 'bar')

#%% 数据分布 --从week1到week19，后面13周开始相较于前面几周开始增多，但充电次数分布都比较均匀
df.week.value_counts()

#%% 
#不同周之间和充电值的关系，根据箱线图表示，所有week的充电值大部分数据都处于非常低的值，在5-12周期间均值有所提升并且大部分数据较高
#但每周都存在一些高于普通充电值的数据，并且每周分布比较均匀，而且这些数据在某一阶段出都现了隔断，并不是持续的值。并且这些值基本没有变化。
plt.figure(figsize=(25,8))
sns.boxplot(x = 'week',y = 'var_value',data = df)
plt.show()

#%% 数据分布
df.var_name.value_counts()

#%% 数据可视化
#不同的版本分布情况较为均匀，基本每个版本的充电桩个数都在6000左右，只有congestion版本要是其他版本的2/3左右
#并且site_size的版本5更少，可能是推出的新品，或者是成本比较昂贵的版本
plt.figure(figsize=(30,8))
df.var_name.value_counts().plot(kind='bar')
plt.show()

#%%数据可视化
#大量版本的充电桩都处于低范围的值，有几个版本有很多处于高范围的值，site——sizev2,和v3,utilization_v3,energy版本的值普遍较高，更为突出的是V6
plt.figure(figsize=(60,15))
sns.boxplot(x = 'var_name',y = 'var_value',data = df)
plt.show()

#下面进行聚类模型构建前的准备
#%%-- 删除无用变量id
data = df.iloc[:,1:]

#%% 这里存在分类变量，所以如果使用简单kmeans分类效果可能不会太理想，把分类变量进行处理
#然后标准化连续变量，使用KPrototypes算法进行聚类
from kmodes.kprototypes import KPrototypes
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat = data[['site_type','week','var_name']]
data_cat = cat.apply(le.fit_transform)

data_num = data[['var_value']]
data_num.var_value = preprocessing.scale(data_num.var_value)

data_cust = pd.concat([data_cat, data_num], ignore_index=True,axis = 1)
#%% 把数据变为矩阵
data_cust_matrix = data_cust.as_matrix()
#%% 选择最合适的K,根据使用MiniBatchKMeans的效果，猜测可能K存在不会很大
#但由于计算机问题，没有运行出结果
cost = []
for num_clusters in list(range(1,20)):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
    kproto.fit_predict(data, categorical=[0,1,2])
    cost.append(kproto.cost_)

plt.plot(cost)

#%% 分类结果结合到数据集上
kproto = KPrototypes(n_clusters=k, init='Cao')
clusters = kproto.fit_predict(data_cust_matrix, categorical=[0,1,2])
data['cluster'] = clusters

#%% 分类结果进行可视化
datacluster = pd.DataFrame(data['cluster'].value_counts())
sns.barplot(x=datacluster.index, y=datacluster['cluster'])

#%% 把数据帧输出为csv文件
data.to_csv("result1.csv",index=False)