import torch
import matplotlib.pyplot as plt
from MODEL import *
import pandas as pd
import numpy as np
import seaborn as sns

model = Deep_CNN()
model.load_state_dict(torch.load('deep_cnn_regression_cosh.pkl'),strict=False)
model.to('cuda')

data1 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/248114222.txt',delimiter=',', skiprows=0)
data2 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/67804123.txt',delimiter=',', skiprows=0)
data3 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/67804129.txt',delimiter=',', skiprows=0)
data4 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/371410113.txt',delimiter=',', skiprows=0)
data5 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/504401133.txt',delimiter=',', skiprows=0)
data6 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/505301116.txt',delimiter=',', skiprows=0)
data7 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/107915114.txt',delimiter=',', skiprows=0)
data8 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/257114202.txt',delimiter=',', skiprows=0)
data9 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/364307053.txt',delimiter=',', skiprows=0)
data10 = np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/582116194.txt',delimiter=',', skiprows=0)

lis = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
y = ['A5', 'A5', 'A5', 'F0', 'F0', 'F0', 'A2', 'G5', 'F0', 'A5']
scores = ['k4', 'G7', 'G7', 'G4', 'G3', 'G2', 'F3', 'K5', 'G0', 'F5']
for i in range(len(lis)):
    plt.subplot(5,2,i+1)
    plt.title('Label:%s Pred:%s'%(y[i],scores[i]))
    plt.plot(lis[i])
plt.tight_layout()
plt.show()