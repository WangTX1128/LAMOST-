import numpy as np
import torch.nn as nn
import torch
from MODEL import Deep_CNN
from function import *
from torch.utils.data import Dataset, TensorDataset
import multiprocessing
from time import time
from tqdm import tqdm
import os

def lodding_map(idx):
    path = 'D:\PycharmProjects\DataProcess\jocbi_AFGK/'
    data = np.loadtxt(path+str(idx),delimiter=',', skiprows=0).astype('float16')
    data = np.column_stack((data.reshape(1,-1),np.zeros((1,10))))
    obsid = idx.split('.')[0].split('_')[1]
    data = np.append(data, int(obsid), axis=None)
    return data

def load_data_GPU():
    path = 'D:\PycharmProjects\DataProcess\jocbi_AFGK/'
    data_list = []
    time1 = time()
    print('Loading data...................')
    for i in os.listdir(path):
        if os.path.splitext(i)[1] == '.txt':
            data_list.append(i)

    pool = multiprocessing.Pool(6)
    data_frame = np.array(pool.map(lodding_map,tqdm(data_list,position=0, leave=True, ncols=75, ascii=True)))
    pool.close()
    pool.join()
    print('-----------------Parallel Done------------------')
    tlabel = data_frame[:,0]
    data = data_frame[:,1:]
    data_set = TensorDataset(torch.from_numpy(data,), torch.from_numpy(tlabel))
    time2 = time()
    print(time2 - time1)
    return  data_set


def test_net(epoch, model, test_loader, test=False):
    test_loss = 0.0
    correct = 0
    pred_list = torch.zeros(1,1,dtype=torch.float).to('cuda')#网络预测值
    true_list = torch.zeros(1,1,dtype=torch.float).to('cuda')#真实值
    name_array = np.zeros((1,1))
    model.eval()
    with torch.no_grad():
        prefetcher = data_prefetcher(test_loader, test)
        if test == True:
            data, label, data_name = prefetcher.next()
        else:
            data, label = prefetcher.next()
        batch_idx = 0
        while data is not None:
            batch_idx += 1

            output = model(data)  #batch * class
            test_loss += criterion(output.squeeze(1), label).item()  # 将一批的损失相加
            if test == True:
            #    error_name(output, data_name, pred.eq(label.view_as(pred)))
                name_array = np.column_stack((name_array, data_name))
            pred_list = torch.cat((pred_list,output.view(1,-1)), 1)
            true_list = torch.cat((true_list,label.view(1,-1)),1)
            '''
            if (batch_idx+1) % 50 == 0:
                print('[{}/{} ({:.0f}%)]'.format(batch_idx*len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))
            '''
            if test == True:
                data, label, data_name = prefetcher.next()
            else:
                data, label = prefetcher.next()
    test_loss /= len(test_loader.dataset)
    
    MAE_error, max_error, std_error = MAE(epoch, pred_list[:,1:], true_list[:,1:], plt_flag=True, data_name=name_array[:,1:])
    pred_label_plot(pred_list[:,1:], true_list[:,1:])
    print('\nTest set: Average loss: {:.4f}, MAE_error / max_error: {:.2f}/{:.2f}, Std:{:.2f}\n'.format(
        test_loss, MAE_error, max_error, std_error))

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True       
    jocbi_data = load_data_GPU()
    test_loader = torch.utils.data.DataLoader(jocbi_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    model = Deep_CNN()
    model.load_state_dict(torch.load('deep_cnn_regression_cosh.pkl'))
    model.cuda()
    criterion = logcosh()
    test_net(1, model,test_loader, test=True)
