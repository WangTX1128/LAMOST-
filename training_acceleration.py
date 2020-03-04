import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, TensorDataset

import multiprocessing
from time import time
from tqdm import tqdm
'''

本程序讲所有数据读入GPU内存，请确保GPU内存在满足模型训练的情况下，仍然足以装下所有数据

'''

def lodding_map_train(idx):
    #path = '/root/train_data/'
    path = r'D:\PycharmProjects\DataProcess\type_id_7类数据集\train_data/'
    data = np.loadtxt(path+str(idx),delimiter=',', skiprows=0).astype('float16')
    return data

def lodding_map_valid(idx):
    #path = '/root/valid_data/'
    path = r'D:\PycharmProjects\DataProcess\type_id_7类数据集\valid_data/'
    data = np.loadtxt(path+str(idx),delimiter=',', skiprows=0).astype('float16')
    obsid = idx.split('.')[0]
    data = np.append(data, int(obsid), axis=None)
    return data

def lodding_map_test(idx):
    #path = '/root/test_data/'
    path = r'D:\PycharmProjects\DataProcess\type_id_7类数据集\test_data/'
    data = np.loadtxt(path+str(idx),delimiter=',', skiprows=0).astype('float16')
    obsid = idx.split('.')[0]
    data = np.append(data, int(obsid), axis=None)
    return data

def load_data_GPU(task):
    #path = '/root/%s_data/'%task
    path = 'D:\PycharmProjects\DataProcess/type_id_7类数据集/%s_data/'%task
    data_list = []
    time1 = time()
    print('Loading data...................')
    for i in os.listdir(path):
        if os.path.splitext(i)[1] == '.txt':
            data_list.append(i)

    pool = multiprocessing.Pool(6)
    if task == 'train':
        data_frame = np.array(pool.map(lodding_map_train,tqdm(data_list,position=0, leave=True, ncols=75, ascii=True)))
    if task =='valid':
        data_frame = np.array(pool.map(lodding_map_valid, tqdm(data_list, position=0, leave=True, ncols=75, ascii=True)))
    if task == 'test':
        data_frame = np.array(pool.map(lodding_map_test, tqdm(data_list, position=0, leave=True, ncols=75, ascii=True)))

    pool.close()
    pool.join()
    print('-----------------Parallel Done------------------')
    tlabel = data_frame[:,0]#带光谱次型
    #teff = data_frame[:,1]#温度
    #type = data_frame[:,2]#光谱型
    data = data_frame[:,1:]#使用了新数据集
    data_set = TensorDataset(torch.from_numpy(data,), torch.from_numpy(tlabel))
    time2 = time()
    print(time2 - time1)
    return  data_set

def load_data_all(task):
    data_frame = np.loadtxt(task+'.txt', delimiter=',').reshape(-1,3749)
    data = data_frame[:,1:]
    target = data_frame[:,0]
    print(target)
    data_frame = TensorDataset(torch.from_numpy(data),torch.from_numpy(target))

    return data_frame


class spectrum_data_GPU(Dataset):
    def __init__(self, data_index, data_mapping, data,transform=None):
        self.data_index = data_index
        self.transform = transform
        #self.path = 'D:\PycharmProjects\stellar_data/'
        self.path = '/root/data/stellar_data/'
        self.data_mapping = data_mapping
        self.data = data

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        data_name = self.data_index.loc[item, 'obsid'].astype(int)
        #data = np.loadtxt(self.path+str(data_name)+'.txt',delimiter=',', skiprows=0).astype('float32')
        idx = self.data_mapping[self.data_mapping['name'] == data_name].index.tolist()
        if idx:
            data_item = self.data[idx]
        else:
            print('expect:%d'%data_name+' but got none')
            print('loss data!!')
            raise TypeError
        label = self.data_index.loc[item, 'label']

        return data_item, label

if __name__ == '__main__':
    load_data_GPU('valid')



