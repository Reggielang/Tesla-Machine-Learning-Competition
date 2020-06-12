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
my_array = ['2019-01-15','2019-02-15','2019-03-15','2019-04-15',
             '2019-05-15','2019-06-15','2019-07-15','2019-08-15',
             '2019-09-30','2019-10-15','2019-11-15','2019-12-15']

df['date'] = my_array

data = df
data.iloc[:,0:25].astype(int)
data['date']=pd.to_datetime(data['date'],format='%Y-%m-%d')
#%% 观察了一下这些城市的数据大小情况
print(data.iloc[:,0:25].agg(['min','mean','median','max','std']))

#%% --在绘图中显示中文 --查看每月不同城市的数据趋势情况
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

data.iloc[:,0:25].plot(figsize=(20,10),fontsize=20)
plt.title('不同城市的限售数据曲线',fontsize=20)
plt.xlabel('不同月份',fontsize=20)
plt.ylabel('不同城市',fontsize=20)
plt.show()

#%%
data_1 = data[['Beijing','date']]
data_1 = data_1.drop(columns='date')


#%%
#%%
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model
 

#%%%
# 创建数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

#一个城市的模型
data_1 = data['Beijing']
 
if __name__ == '__main__':
    # 加载数据
    data_1 = data['Beijing']
    dataset = data_1.values.reshape(-1, 1)
    # 将整型变为float
    dataset = dataset.astype('float32')     
    
    # 数据处理，归一化至0~1之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    # 创建测试集和训练集
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)           #单步预测
    testX, testY = create_dataset(test, look_back)
    
    # 调整输入数据的格式
    trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))       #（样本个数，1，输入的维度）
    testX = numpy.reshape(testX, (testX.shape[0], look_back, testX.shape[1]))
    
    # 创建LSTM神经网络模型
    model = Sequential()
    model.add(LSTM(120, input_shape=(trainX.shape[1], trainX.shape[2])))            #输入维度为1，时间窗的长度为1，隐含层神经元节点个数为120
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    
    # 预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # 反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    # 计算得分
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    # 绘图
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)