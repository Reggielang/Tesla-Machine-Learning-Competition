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
a = df[df.var_value>1000]
#df = df.drop(df[df.var_value>1000].index)
#%%
b = df[df.var_value<1]
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

#%% 生成哑变量
data = pd.concat([data,pd.get_dummies(df['site_type']),pd.get_dummies(df['week']),pd.get_dummies(df['var_name'])],axis = 1)

#%%
#一般会保留K-1个哑变量
del data['site_type']
del data['week']
del data['var_name']
del data['c']
del data['week_19']
del data['utilization_v2']

#%% 进行算法之前要进行标准化！
data.info()
from sklearn import preprocessing
data.var_value = preprocessing.scale(data.var_value)

#%% 因为数据处于非常大的样本，所以需要进行Minnibatch kmeans算法处理
#这里采用了手肘法选取最佳K值
#分群效果可以从图中看出
from sklearn.cluster import  MiniBatchKMeans, KMeans  
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
batch_size = 1000
  

SSE = []  # 存放每次结果的误差平方和
for k in range(1,50):
    km = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=batch_size, random_state=50)  # 构造聚类器
    km.fit(data)
    SSE.append(km.inertia_)

X = range(1,50)
# 绘制K的个数与SSE的关系
plt.figure(figsize=(30,10))
plt.plot(X,SSE,'o-')
plt.xlabel('聚类个数')
plt.ylabel('簇内离差平方和')
plt.title('选择最优的聚类个数')
plt.show()
#%% 
#对于上面的图中显示，K大约在10到20之间，簇的方差开始缓慢减少。所以K值取在这里比较合适
km = MiniBatchKMeans(init='k-means++', n_clusters=15, batch_size=batch_size, random_state=100)
result = km.fit(data)

#%% 把聚类结果标记在原来的数据集上
data_l=df.join(pd.DataFrame(result.labels_))
data_l=data_l.rename(columns={0: "cluster"})
data_l.head()

data_l.cluster.value_counts().plot(kind = 'bar')

#%% 把数据帧输出为csv文件
data_l.to_csv("task1.csv",index=False)
