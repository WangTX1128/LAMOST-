import numpy as np
import training_acceleration
import multiprocessing
from time import time
from tqdm import tqdm

train, train_target = training_acceleration.load_data_GPU('train') #修改源码，返回两个数组
test , test_target= training_acceleration.load_data_GPU('test')
test = np.column_stack((test,test_target))
def predding(x):
    x = x[:-1]
    tt = x[-1]
    #到某个训练样本的平方距离，dist存储到所有训练样本的平方距离
    d = (np.square(x - train)).sum(axis=1)
    h = d.min()#最小平方距离
    if h == 0:
        indx = d.index(h)
        pred = train_target[index]
    else:
        part = np.exp(d/h * (-1/2))
        upp = np.dot(part,train_target)
        dow = np.sum(part)
        pred = upp/dow
    preded = np.array([pred,tt])
    return preded

pool = multiprocessing.Pool(16)
preded = np.array(pool.map(predding,tqdm(test,position=0, leave=True, ncols=75, ascii=True)))
pool.close()
pool.join()
print(preded.shape)
np.savetxt('predd.txt',preded, delimiter=',', fmt='%.16f')
error = abs(preded[:,0] - preded[:,1])
n = 0
for i in error:
    if i <= 0.2:
        n += 1
print('0.2个误差以内：%d'%n)
max_error = max(error)
print('最大绝对误差：%f'%max_error)
MAE_error = (error.sum() - max_error) / (error.shape[1] - 1)
print('平均绝对误差：%f'%MAE_error)
std_error = np.std(error)
print('标准差:%f'%std_error)