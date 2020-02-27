import shap
import MODEL
import torch
import numpy as np
import matplotlib.image as mpimg
from function import *
from torch.utils.data import Dataset, TensorDataset
import multiprocessing
from time import time
from tqdm import tqdm
import os

model = MODEL.Deep_CNN()
model.load_state_dict(torch.load('best_deep_cnn_classification.pkl'),strict=False)
model.eval()

def lodding_map(idx):
    path = 'D:\PycharmProjects\DataProcess/4类数据集/regression\BACKGROUND/'
    data = np.loadtxt(path+str(idx),delimiter=',', skiprows=0).astype('float16')
    #data = np.column_stack((data.reshape(1,-1),np.zeros((1,10))))
    #obsid = idx.split('.')[0].split('_')[1]
    #data = np.append(data, int(obsid), axis=None)
    return data

def load_data_GPU():
    path = 'D:\PycharmProjects\DataProcess/4类数据集/regression\BACKGROUND/'
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
    ttype = data_frame[:,2]
    data = data_frame[:,3:]
    #data_set = TensorDataset(torch.from_numpy(data,), torch.from_numpy(ttype))
    data_set = torch.from_numpy(data)
    time2 = time()
    print(time2 - time1)
    return  data_set.unsqueeze(1)


if __name__ == "__main__":
        
    A1 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/A/803222.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    A2 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/A/902029.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    F1 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/F/606239.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    F2 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/F/803162.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    G1 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/G/601041.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    G2 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/G/602142.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    K1 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/K/107103.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    K2 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/K/202065.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)

    X_tensor = torch.cat([A1, A2], dim=0)
    X_tensor = torch.cat([X_tensor, F1], dim=0)
    X_tensor = torch.cat([X_tensor,F2], dim=0)
    X_tensor = torch.cat([X_tensor, G1], dim=0)
    X_tensor = torch.cat([X_tensor, G2], dim=0)
    X_tensor = torch.cat([X_tensor,K1], dim=0)
    X_tensor = torch.cat([X_tensor, K2], dim=0).unsqueeze(1)
    print(X_tensor.shape)

    data_set = load_data_GPU()
    to_explain = X_tensor #要解释的样本
    y_tensor = torch.Tensor([0, 1, 2, 3])

    picA1 = mpimg.imread('pic\A1_803222.png')
    picA2 = mpimg.imread('pic\A2_902029.png')
    picF1 = mpimg.imread('pic\F1_606239.png')
    picF2 = mpimg.imread('pic\F2_803162.png')
    picG1 = mpimg.imread('pic\G1_601041.png')
    picG2 = mpimg.imread('pic\G2_602142.png')
    picK1 = mpimg.imread('pic\K1_107103.png')
    picK2 = mpimg.imread('pic\K2_202065.png')

    pic_list = [picA1,picA2,picF1,picF2,picG1,picG2,picK1,picK2]

    e1 = shap.GradientExplainer((model, model.conv1), data_set.float(),local_smoothing=0)
    shap_values1,indexes1 = e1.shap_values(to_explain, ranked_outputs=1, nsamples=2000)

    #e2 = shap.GradientExplainer((model, model.layer3), data_set.float(),local_smoothing=0.5)
    #shap_values2,indexes2 = e2.shap_values(to_explain, ranked_outputs=1, nsamples=2000)

    #e3 = shap.GradientExplainer((model, model.layer5), data_set.float(),local_smoothing=0.5)
    #shap_values3,indexes3 = e3.shap_values(to_explain, ranked_outputs=1, nsamples=2000)

    # get the names for the classes
    class_names = ['A','F','G','K']
    index_names = np.vectorize(lambda x: class_names[x])(indexes1)
    shap_values1 = [np.expand_dims(s,axis=2) for s in shap_values1]
    #shap_values2 = [np.expand_dims(s,axis=2) for s in shap_values2]
    #shap_values3 = [np.expand_dims(s,axis=2) for s in shap_values3]

    # plot the explanations
    shap_values1 = [s.transpose((0,2,3,1)) for s in shap_values1]# samples x h x w x channels
    #shap_values2 = [s.transpose((0,2,3,1)) for s in shap_values2]
    #shap_values3 = [s.transpose((0,2,3,1)) for s in shap_values3]
    print(len(shap_values1))
    print(shap_values1[0].shape)
    to_explain = np.swapaxes(np.expand_dims(to_explain,axis=2),2,3)
    
    shap.image_plot(shap_values1, to_explain, pic_list=pic_list, labels=index_names)
    