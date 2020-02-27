import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from training_acceleration import *
from BiLSTM import *
from function import *

writer = SummaryWriter('runs/lstm')




class spectrum_data(Dataset):
    def __init__(self, data_index, transform=None):
        self.data_index = data_index
        self.transform = transform
        self.path = 'D:\PycharmProjects\stellar_data/'
        #self.path = '/root/data/stellar_data/'

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        data_name = self.data_index.loc[item, 'obsid']#.astype(int)
        data = np.loadtxt(self.path+str(data_name)+'.txt',delimiter=',', skiprows=0)[:3748].astype('float32')
        label = self.data_index.loc[item, 'label']

        return data, label




def train_net(epoch, model, data_trainer, criterion, optimizer):
    running_loss = 0.0
    correct = 0
    train_loss = 0.0
    pred_list = torch.zeros(1, 1, dtype=torch.long).to('cuda')  # 网络预测值
    true_list = torch.zeros(1, 1, dtype=torch.long).to('cuda')  # 真实值
    model.train()
    #for batch_idx, (data ,label) in tqdm(enumerate(data_trainer)):
    prefetcher = data_prefetcher(data_trainer)
    data, label = prefetcher.next()
    batch_idx = 0
    while data is not None:
        batch_idx += 1
        #.reshape(-1, 937, 4)
        if torch.any(torch.isnan(data)) | torch.any(torch.isnan(label)):
            break

        optimizer.zero_grad()
        output = model(data)

        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        pred_list = torch.cat((pred_list, pred.view(1, -1)), 1)
        true_list = torch.cat((true_list, label.view(1, -1)), 1)
        correct += pred.eq(label.view_as(pred)).sum().item()
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 15)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
        optimizer.step()
        # scheduler.step()

        running_loss += loss.item()
        
        if  batch_idx  % 50 == 0:
            writer.add_scalar('training loss',
                              running_loss / 50,
                              (epoch - 1) * len(data_trainer) + batch_idx)
            print('Train epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                          len(data_trainer.dataset),
                                                                          100. * batch_idx / len(data_trainer),
                                                                          running_loss / 50))
            running_loss = 0.0

        data, label = prefetcher.next()
    print("Training Finished")
    true_list = true_list.squeeze().long().to('cpu').tolist()
    pred_list = pred_list.squeeze().long().to('cpu').tolist()
    A_error, F_error, G_error, K_error = accuracy_of_label(true_list, pred_list)
    print(A_error, F_error, G_error, K_error)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(data_trainer.dataset), correct, len(data_trainer.dataset),
        100. * correct / len(data_trainer.dataset)))

def test_net(epoch, model, test_loader, test=False):
    test_loss = 0.0
    correct = 0
    pred_list = torch.zeros(1,1,dtype=torch.long).to('cuda')#网络预测值
    true_list = torch.zeros(1,1,dtype=torch.long).to('cuda')#真实值
    model.eval()
    with torch.no_grad():
        #for batch_idx,(data, label) in tqdm(enumerate(test_loader)):
        prefetcher = data_prefetcher(test_loader)
        data, label = prefetcher.next()
        batch_idx = 0
        while data is not None:
            batch_idx += 1

            output = model(data)
            test_loss += criterion(output, label).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标

            pred_list = torch.cat((pred_list,pred.view(1,-1)), 1)
            true_list = torch.cat((true_list,label.view(1,-1)),1)
            correct += pred.eq(label.view_as(pred)).sum().item()
            '''
            if  batch_idx % 50 == 0:
                print('[{}/{} ({:.0f}%)]'.format(batch_idx*len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))
            '''
            data, label = prefetcher.next()

    test_loss /= len(test_loader.dataset)
    if test == False:
        writer.add_scalar('valid loss',
                          test_loss,
                          epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    true_list = true_list.squeeze().long().to('cpu').tolist()
    pred_list = pred_list.squeeze().long().to('cpu').tolist()
    F1_score = f1_score(true_list, pred_list, average='macro')

    writer.add_scalar('valid F1 Score',
                      F1_score,
                      epoch )

    A_error, F_error, G_error, K_error = accuracy_of_label(true_list, pred_list)
    print(A_error, F_error, G_error, K_error)
    valid_F1_score_list.append(F1_score)
    print('\nvalid_F1_score = {:.4f}\n'.format(F1_score))




if __name__ == '__main__':
    learning_rate = 0.00001
    epoch = 5000
    Frozen_Layers = False
    torch.backends.cudnn.benchmark = True

    train_set = load_data_GPU('train')
    print(len(train_set))
    valid_set = load_data_GPU('valid')
    print(len(valid_set))

    valid_F1_score_list = []
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=10, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=10, pin_memory=True)

    #model = Rnn(4, 16, 8, 4)#定义数据为1302行3列,网络隐藏层为三层
    model = BiLSTM(4, 4, 64, 5, bidirectional=True)  # input_size, output_size, hidden_dim, n_layers
    init_lstm(model)
    for i in model.modules():
        weight_init(i)
    #init_lstm(model)
    device = torch.device('cuda')
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 3.774, 14.49, 1]))
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)### 要想使用GPU，此处必须使用cuda()
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0)

    bias_list = (param for name, param in model.named_parameters() if 'bias' in name)
    others_list = (param for name, param in model.named_parameters() if 'bias' not in name)
    parameters = [{'params': bias_list, 'weight_decay': 0},
                  {'params': others_list}]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    model.to(device)

    #加载LSTM预训练参数
    #pretrained_params = torch.load('best_model.pkl')
    #model.load_state_dict(pretrained_params, strict=False)
    model.load_state_dict(torch.load('best_model_lstm.pkl'))
    # 固定LSTM参数
    if Frozen_Layers == True:
        train_layer = ['lstm.weight_ih_l6', 'lstm.weight_hh_l6', 'lstm.bias_ih_l6', 'lstm.bias_hh_l6',
                       'lstm.weight_ih_l7','lstm.weight_hh_l7','lstm.bias_ih_l7','lstm.bias_hh_l7',
                       'classifier.weight', 'classifier.bias']
        for k, v in model.named_parameters():
            if k not in train_layer:
                print(k)
                v.requires_grad = False
        print('LSTM FROZENED!')
        for k, v in model.named_parameters():
            if v.requires_grad == True:
                print('training:',k)

    else:
        print('-------train all model-------')
    max_f1 = 0.85

    for i in range(300, epoch+1):

        train_net(i, model, train_loader, criterion, optimizer)

        test_net(i, model, valid_loader)
        if valid_F1_score_list[-1] > max_f1:
            max_f1 = valid_F1_score_list[-1]
            torch.save(model.state_dict(), 'best_model_lstm.pkl')  # save only the parameters

    model.load_state_dict(torch.load('best_model_lstm.pkl'))
    test_set = load_data_GPU('test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=10, pin_memory=True)
    test_net(1, model,test_loader, test=True)
