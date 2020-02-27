import numpy as np
import training_acceleration
import multiprocessing
from time import time
from tqdm import tqdm

train, train_target = training_acceleration.load_data_GPU('train') #修改源码，返回两个数组
test , test_target= training_acceleration.load_data_GPU('test')
test = np.column_stack((test,test_target))
def predding(data):
    x = data[:-1]
    tt = data[-1]
    #到某个训练样本的平方距离，dist存储到所有训练样本的平方距离
    d = (np.square(x - train)).sum(axis=1)
    h = d.min()#最小平方距离
    if h == 0:
        indx = d.tolist().index(h)
        pred = train_target[index]
    else:
        part = np.exp(d/h * (-1/2))
        upp = np.dot(part,train_target)
        dow = np.sum(part)
        pred = upp/dow
    preded = np.array([pred,tt])
    return preded

time1 = time()
array1 = test[:3000,:]
array2 = test[3000:6000,:]
array3 = test[6000:9000,:]
array4 = test[9000:12000,:]
array5 = test[12000:15000,:]
array6 = test[15000:,:]
size = array1.shape[0]+array2.shape[0]+array3.shape[0]+array4.shape[0]+array5.shape[0]+array6.shape[0]
print(size)
list_ = [array1,array2,array3,array4,array5,array6]
list_2 = []
for arr in list_:
    pool = multiprocessing.Pool()
    preded = np.array(pool.map(predding,tqdm(arr,position=0, leave=True, ncols=75, ascii=True)))
    pool.close()
    pool.join()
    list_2.append(preded)
all_pred = np.concatenate(list_2,axis=0)
np.savetxt('predd.txt',all_pred, delimiter=',', fmt='%.16f')
error = abs(all_pred[:,0] - all_pred[:,1])
n = 0
for i in error:
    if i <= 0.2:
        n += 1
print('0.2个误差以内：%d'%n)
max_error = max(error)
print('最大绝对误差：%f'%max_error)
MAE_error = (error.sum() - max_error) / (error.shape[0] - 1)
print('平均绝对误差：%f'%MAE_error)
std_error = np.std(error)
print('标准差:%f'%std_error)

time2 = time()
print('time:',time2 - time1)