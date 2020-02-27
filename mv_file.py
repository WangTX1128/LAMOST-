import numpy as np
import os
import shutil
import multiprocessing

origin_path = 'D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/'

A_NUM = 500
F_NUM = 500
G_NUM = 500
K_NUM = 500

def get():
    data_list = []
    for i in os.listdir(origin_path):
        if os.path.splitext(i)[1] == '.txt':
            data_list.append(i)
            pass
        pass
    return data_list

def mv_file(data):
    #定义一个根据类别来移动样本的函数
    global A_NUM,F_NUM,G_NUM,K_NUM
    file = np.loadtxt(origin_path+data,delimiter=',',skiprows=0)
    if file[2] == 0 and A_NUM > 0:
        target_path = 'D:\PycharmProjects\DataProcess/4类数据集/regression\A/'
        shutil.move(origin_path+data,target_path+data)
        A_NUM -= 1
    if file[2] == 1 and F_NUM > 0:
        target_path = 'D:\PycharmProjects\DataProcess/4类数据集/regression\F/'
        shutil.move(origin_path+data,target_path+data)
        F_NUM -= 1
    if file[2] == 2 and G_NUM > 0:
        target_path = 'D:\PycharmProjects\DataProcess/4类数据集/regression\G/'
        shutil.move(origin_path+data,target_path+data)
        G_NUM -= 1
    if file[2] == 3 and K_NUM > 0:
        target_path = 'D:\PycharmProjects\DataProcess/4类数据集/regression\K/'
        shutil.move(origin_path+data,target_path+data)
        K_NUM -= 1

if __name__ == '__main__':
    list_ = get() 
    pool = multiprocessing.Pool()
    pool.map(mv_file,list_)
    pool.close()
    pool.join()