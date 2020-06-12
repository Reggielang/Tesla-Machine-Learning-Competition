# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:32:47 2020

@author: REGGIE
"""
#%%$
import numpy as np
import pandas as pd
import seaborn as sns
import os 

os.chdir(r'C:\Users\REGGIE\Desktop\Machine Learning Model Group')
df = pd.read_csv('sales.csv',header=None)
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#%% 通过观察数据可以得出这是一个周期性的时间序列数据
#city = df[0]
#columns = city.drop(index =0)

#%% 对数据进行变动, 并且把每月的123数据假设为每月的月中15号。
df= df.drop(index=0)
df = df.T
df.columns = df.iloc[0,:]
df = df.drop(index=0)
data = df
data.iloc[:,0:25].astype(int)
#%% 观察了一下这些城市的数据大小情况
print(data.iloc[:,0:25].agg(['min','mean','median','max','std']))

#%% --在绘图中显示中文 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
#--查看每月不同城市的数据趋势情况
data.iloc[:,0:25].plot(figsize=(20,10),fontsize=20)
plt.title('不同城市的限售数据曲线',fontsize=20)
plt.xlabel('不同月份',fontsize=20)
plt.ylabel('不同城市',fontsize=20)
plt.show()

#%%
data_1 = data['Beijing']

#%% --画出自相关性和偏相关性图，并做平稳性检验
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = data_1.values.astype(float)
plot_acf(data) #--自相关
plt.savefig('acf.png')
plot_pacf(data) #--偏相关
plt.savefig('pacf.png')
plt.show()

#%% --由检验结果可知原始序列是平稳序列--所以不用进行差分处理
from statsmodels.tsa.stattools import adfuller as ADF
print('原始序列的ADF检验结果为：', ADF(data_1))
#%%
#差分后的结果
D_data_1 = data_1.diff().dropna()
print("差分序列的ADF 检验结果为", ADF(D_data_1))

#%% 一阶差分可视化
data = D_data_1.values.astype(float)
plot_acf(data) #--自相关
plt.savefig('acf.png')
plot_pacf(data) #--偏相关
plt.savefig('pacf.png')
plt.show()
#%% 二阶差分
#D_data_1 =D_data_1.diff().dropna()
#print("差分序列的ADF 检验结果为", ADF(D_data_1))

#%%
#白噪声检验
#from statsmodels.stats.diagnostic import acorr_ljungbox
#返回统计量和p值
#print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data_1.astype(float), lags=1))

#%% 二阶差分可视化
#D_data_1 = D_data_1.values.astype(float)
#plot_acf(data) #--自相关
#plt.savefig('acf.png')
#plot_pacf(data) #--偏相关
#plt.savefig('pacf.png')
#plt.show()

#%%
#定阶
#从一阶差分后的序列是平稳的非白噪声序列可以看出ARIMA模型中的d=1

from statsmodels.tsa.arima_model import ARIMA
pmax = int(len(data_1)/10) #一般阶数不超过length/10
qmax = int(len(data_1)/10) #一般阶数不超过length/10
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):       
        try:
            #存在部分报错，所以用try来跳过报错。
            tmp.append(ARIMA(data, (p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(p,q)
model = ARIMA(data,(p,1,q)).fit()
model.summary2()
model.forecast(6)[0]

#%% --对定阶前的处理
columns = df.columns.tolist()
test = {}
for i in columns:
    city = df[i].values.astype(float)
    print(i,city)
    test[i] = city

#%%
#从一阶差分后的序列是平稳的非白噪声序列可以看出ARIMA模型中的d=1
from statsmodels.tsa.arima_model import ARIMA
for k,i in test.items():
    data = i
    pmax = int(len(data_1)/10) #一般阶数不超过length/10
    qmax = int(len(data_1)/10) #一般阶数不超过length/10
    bic_matrix = [] #bic矩阵
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                #存在部分报错，所以用try来跳过报错。
                tmp.append(ARIMA(data, (p,1,q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
    p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
    print(k,p,q)

      
#%%    
#确定了ARIMA模型的三个参数就可以构建模型
#这里确实没找到合适的方法来进行自动化
#只能一个城市一个城市的进行预测
#预测返回的结果是为期6天的预测，返回了结果，标准误差，置信区间。 
#%%
#Beijing --0 1 1
model_Beijing = ARIMA(test['Beijing'],(0,1,1)).fit()
print(model_Beijing.summary2(),'\n',model_Beijing.forecast(6)[0])
list_1 = pd.DataFrame(model_Beijing.forecast(6)[0])

#%%
#Tianjin --0 1 1
model_Tianjin = ARIMA(test['Tianjin'],(0,1,1)).fit()
print(model_Tianjin.summary2(),'\n',model_Tianjin.forecast(6)[0])
list_2 = pd.DataFrame(model_Tianjin.forecast(6)[0])
#%%
#xi'an --0 1 1
model_xian = ARIMA(test["xi'an"],(0,1,1)).fit()
print(model_xian.summary2(),'\n',model_xian.forecast(6)[0])
list_3 = pd.DataFrame(model_xian.forecast(6)[0])
#%%
#Qingdao--0 1 1
model_Qingdao = ARIMA(test['Qingdao'],(0,1,1)).fit()
print(model_Qingdao.summary2(),'\n',model_Qingdao.forecast(6)[0])
list_4 = pd.DataFrame(model_Qingdao.forecast(6)[0])
#%%
#Shenyang --0 1 1
model_Shenyang = ARIMA(test['Shenyang'],(0,1,1)).fit()
print(model_Shenyang.summary2(),'\n',model_Shenyang.forecast(6)[0])
list_5 = pd.DataFrame(model_Shenyang.forecast(6)[0])
#%%
#Zhengzhou--1 1 1
model_Zhengzhou = ARIMA(test['Zhengzhou'],(1,1,1)).fit()
print(model_Zhengzhou.summary2(),'\n',model_Zhengzhou.forecast(6)[0])
list_6 = pd.DataFrame(model_Zhengzhou.forecast(6)[0])
#%%
#Dalian --1 1 1
model_Dalian = ARIMA(test['Dalian'],(1,1,1)).fit()
print(model_Dalian.summary2(),'\n',model_Dalian.forecast(6)[0])
list_7 = pd.DataFrame(model_Dalian.forecast(6)[0])
#%%
#Shanghai --0 1 1
model_Shanghai = ARIMA(test['Shanghai'],(0,1,1)).fit()
print(model_Shanghai.summary2(),'\n',model_Shanghai.forecast(6)[0])
list_8 = pd.DataFrame(model_Shanghai.forecast(6)[0])
#%%
#Hangzhou --0 1 1
model_Hangzhou = ARIMA(test['Hangzhou'],(0,1,1)).fit()
print(model_Hangzhou.summary2(),'\n',model_Hangzhou.forecast(6)[0])
list_9 = pd.DataFrame(model_Hangzhou.forecast(6)[0])
#%%
#Suzhou--0 1 1
model_Suzhou = ARIMA(test['Suzhou'],(0,1,1)).fit()
print(model_Suzhou.summary2(),'\n',model_Suzhou.forecast(6)[0])
list_10 = pd.DataFrame(model_Suzhou.forecast(6)[0])
#%%
#Nanjing --0 1 1
model_Nanjing = ARIMA(test['Nanjing'],(0,1,1)).fit()
print(model_Nanjing.summary2(),'\n',model_Nanjing.forecast(6)[0])
list_11 = pd.DataFrame(model_Nanjing.forecast(6)[0])
#%%
#ningbo --1 1 1
model_ningbo = ARIMA(test['ningbo'],(1,1,1)).fit()
print(model_ningbo.summary2(),'\n',model_ningbo.forecast(6)[0])
list_12 = pd.DataFrame(model_ningbo.forecast(6)[0])
#%%
#Changzhou--0 1 1
model_Changzhou = ARIMA(test['Changzhou'],(0,1,1)).fit()
print(model_Changzhou.summary2(),'\n',model_Changzhou.forecast(6)[0])
list_13 = pd.DataFrame(model_Changzhou.forecast(6)[0])
#%%
#wuhan --0 1 1
model_wuhan = ARIMA(test['wuhan'],(0,1,1)).fit()
print(model_wuhan.summary2(),'\n',model_wuhan.forecast(6)[0])
list_14 = pd.DataFrame(model_wuhan.forecast(6)[0])
#%%
#Wenzhou --0 1 0
model_Wenzhou = ARIMA(test['Wenzhou'],(0,1,0)).fit()
print(model_Wenzhou.summary2(),'\n',model_Wenzhou.forecast(6)[0])
list_15 = pd.DataFrame(model_Wenzhou.forecast(6)[0])
#%%
#Guangzhou --0 1 1
model_Guangzhou = ARIMA(test['Guangzhou'],(0,1,1)).fit()
print(model_Guangzhou.summary2(),'\n',model_Guangzhou.forecast(6)[0])
list_16 = pd.DataFrame(model_Guangzhou.forecast(6)[0])
#%%
#Changsha --1 1 0
model_Changsha = ARIMA(test['Changsha'],(1,1,0)).fit()
print(model_Changsha.summary2(),'\n',model_Changsha.forecast(6)[0])
list_17 = pd.DataFrame(model_Changsha.forecast(6)[0])
#%%
#Nanning --0 1 1
model_Nanning = ARIMA(test['Nanning'],(0,1,1)).fit()
print(model_Nanning.summary2(),'\n',model_Nanning.forecast(6)[0])
list_18 = pd.DataFrame(model_Nanning.forecast(6)[0])
#%%
#Nanchang --0 1 1
model_Nanchang = ARIMA(test['Nanchang'],(0,1,1)).fit()
print(model_Nanchang.summary2(),'\n',model_Nanchang.forecast(6)[0])
list_19 = pd.DataFrame(model_Nanchang.forecast(6)[0])
#%%
#Shenzhen --0 1 1
model_Shenzhen = ARIMA(test['Shenzhen'],(0,1,1)).fit()
print(model_Shenzhen.summary2(),'\n',model_Shenzhen.forecast(6)[0])
list_20 = pd.DataFrame(model_Shenzhen.forecast(6)[0])
#%%
#Xiamen --0 1 1
model_Xiamen = ARIMA(test['Xiamen'],(0,1,1)).fit()
print(model_Xiamen.summary2(),'\n',model_Xiamen.forecast(6)[0])
list_21 = pd.DataFrame(model_Xiamen.forecast(6)[0])
#%%
#Haikou --0 1 1
model_Haikou = ARIMA(test['Haikou'],(0,1,1)).fit()
print(model_Haikou.summary2(),'\n',model_Haikou.forecast(6)[0])
list_22 = pd.DataFrame(model_Haikou.forecast(6)[0])
#%%
#Dongguan --0 1 1
model_Dongguan = ARIMA(test['Dongguan'],(0,1,1)).fit()
print(model_Dongguan.summary2(),'\n',model_Dongguan.forecast(6)[0])
list_23 = pd.DataFrame(model_Dongguan.forecast(6)[0])
#%%
#Chengdu --0 1 1
model_Chengdu = ARIMA(test['Chengdu'],(0,1,1)).fit()
print(model_Chengdu.summary2(),'\n',model_Chengdu.forecast(6)[0])
list_24 = pd.DataFrame(model_Chengdu.forecast(6)[0])
#%%
#Chongqing --0 1 1
model_Chongqing = ARIMA(test['Chongqing'],(0,1,1)).fit()
print(model_Chongqing.summary2(),'\n',model_Chongqing.forecast(6)[0])
list_25 = pd.DataFrame(model_Chongqing.forecast(6)[0])

#%% --合并
predict =  pd.concat([list_1,list_2,list_3,list_4,list_5,list_6,list_7,
           list_8,list_9,list_10,list_11,list_12,list_13,list_14,
           list_15,list_16,list_17,list_18,list_19,list_20,list_21,
           list_22,list_23,list_24,list_25],axis=1)

#%% -- 改变列名
predict.columns = columns
#%% --合并原始数据和预测数据
data = pd.concat([df,predict],axis=0)

#%% 把数据帧输出为csv文件
data.to_csv("task2.csv",index=False)
