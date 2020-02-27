import torch
import matplotlib.pyplot as plt
from MODEL import *
import pandas as pd
import numpy as np
import seaborn as sns

model = Deep_CNN()
model.load_state_dict(torch.load('best_deep_cnn_classification.pkl'),strict=False)
model.to('cuda')


def compute_saliency_maps(X, y, model):
    """
    X表示图片, y表示分类结果, model表示使用的分类模型
    
    Input : 
    - X : Input images : Tensor of shape (N, 3, H, W)
    - y : Label for X : LongTensor of shape (N,)
    - model : A pretrained CNN that will be used to computer the saliency map
    
    Return :
    - saliency : A Tensor of shape (N, H, W) giving the saliency maps for the input images
    """
    # 确保model是test模式
    model.eval()
    X = X.to('cuda')
    y = y.to('cuda')
    # 确保X是需要gradient
    X.requires_grad_()
    
    saliency = None
    
    scores = model.forward(X)
    #由于是回归模型，输出的scores就是最终预测结果
    pred = scores.max(1, keepdim=True)[1]  # 找到概率最大的下标
    scores = scores.gather(1, y.view(-1, 1).long())#.squeeze(1) # 得到正确分类的得分
    scores.backward(torch.FloatTensor([[1.], [1.], [1.], [1.]]).to('cuda')) # 初始化，长度应该和scores的样本个数相等
    
    saliency = abs(X.grad.data) # 返回X的梯度绝对值大小
    #saliency, _ = torch.max(saliency, dim=1)# 从3个颜色通道中取绝对值最大的那个通道的数值
    
    return saliency, pred

def show_saliency_maps():
    # Convert X and y from numpy arrays to Torch Tensors
    data1 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/220511002.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    data2 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/346012094.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    data3 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/170613193.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    data4 = torch.from_numpy(np.loadtxt('D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/90202011.txt',delimiter=',', skiprows=0).astype('float32')[3:]).reshape(1,-1)
    X_tensor = torch.cat([data1, data2], dim=0)
    X_tensor = torch.cat([X_tensor, data3], dim=0)
    X_tensor = torch.cat([X_tensor,data4], dim=0).unsqueeze(1)
    y_tensor = torch.Tensor([0, 1, 2, 3])
    print(X_tensor.shape)
    # Compute saliency maps for images in X
    saliency, scores = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.cpu().numpy()
    saliency_plot(X_tensor, y_tensor, scores, saliency)
   
def saliency_plot(data, y, scores, maps):
    name = ['A','F','G','K']
    for i in range(data.shape[0]):
        paper = np.zeros((500, 3738))
        plt.subplot(2,4,i+1)
        plt.title('Label:%s Pred:%s'%(name[int(y[i])],name[int(scores[i])]))
        plt.plot(data[i][0])
        for j in range(500):
            paper[j] = maps[i][0]
        plt.subplot(2,4,i+5)
        plt.imshow(paper,cmap=plt.cm.hot)
    plt.show()

show_saliency_maps()