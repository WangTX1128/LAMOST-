import math
from BiLSTM import *
from MODEL import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#######################################################################################################################
# #LSTM初始化函数



def init_lstm(model):


    model.lstm.bias_ih_l0.data[model.lstm.bias_ih_l0.data.shape[0]//4 : 2 * model.lstm.bias_ih_l0.data.shape[0]//4] = torch.ones(model.lstm.bias_ih_l0.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_ih_l1.data[model.lstm.bias_ih_l1.data.shape[0]//4 : 2 * model.lstm.bias_ih_l1.data.shape[0]//4] = torch.ones(model.lstm.bias_ih_l1.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_ih_l2.data[model.lstm.bias_ih_l2.data.shape[0]//4 : 2 * model.lstm.bias_ih_l2.data.shape[0]//4] = torch.ones(model.lstm.bias_ih_l2.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_ih_l3.data[model.lstm.bias_ih_l3.data.shape[0]//4 : 2 * model.lstm.bias_ih_l3.data.shape[0]//4] = torch.ones(model.lstm.bias_ih_l3.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_ih_l4.data[model.lstm.bias_ih_l4.data.shape[0]//4 : 2 * model.lstm.bias_ih_l4.data.shape[0]//4] = torch.ones(model.lstm.bias_ih_l4.data.shape[0]//4, dtype=torch.float)
    #model.lstm.bias_ih_l5.data[model.lstm.bias_ih_l5.data.shape[0]//4 : 2 * model.lstm.bias_ih_l5.data.shape[0]//4] = torch.ones(model.lstm.bias_ih_l5.data.shape[0]//4, dtype=torch.float)

    model.lstm.bias_hh_l0.data[model.lstm.bias_hh_l0.data.shape[0]//4 : 2 * model.lstm.bias_hh_l0.data.shape[0]//4] = torch.ones(model.lstm.bias_hh_l0.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_hh_l1.data[model.lstm.bias_hh_l1.data.shape[0]//4 : 2 * model.lstm.bias_hh_l1.data.shape[0]//4] = torch.ones(model.lstm.bias_hh_l1.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_hh_l2.data[model.lstm.bias_hh_l2.data.shape[0]//4 : 2 * model.lstm.bias_hh_l2.data.shape[0]//4] = torch.ones(model.lstm.bias_hh_l2.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_hh_l3.data[model.lstm.bias_hh_l3.data.shape[0]//4 : 2 * model.lstm.bias_hh_l3.data.shape[0]//4] = torch.ones(model.lstm.bias_hh_l3.data.shape[0]//4, dtype=torch.float)
    model.lstm.bias_hh_l4.data[model.lstm.bias_hh_l4.data.shape[0]//4 : 2 * model.lstm.bias_hh_l4.data.shape[0]//4] = torch.ones(model.lstm.bias_hh_l4.data.shape[0]//4, dtype=torch.float)
    #model.lstm.bias_hh_l5.data[model.lstm.bias_hh_l5.data.shape[0]//4 : 2 * model.lstm.bias_hh_l5.data.shape[0]//4] = torch.ones(model.lstm.bias_hh_l5.data.shape[0]//4, dtype=torch.float)
    print('bias set 1')

    torch.nn.init.normal_(model.fc._parameters['weight'], mean=0, std=1)
    torch.nn.init.normal_(model.fc._parameters['bias'], mean=0, std=1)

#######################################################################################################################
#初始化函数
def weight_init(m):
# 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

    elif isinstance(m, nn.Conv1d):
        n = 1 * m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        print('init %s'%m)
    elif isinstance(m, nn.BatchNorm1d):
# m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.LSTM):
        for key in m._parameters.keys():
            if 'weight' in key:
                print('init %s'%key)
                torch.nn.init.orthogonal_((m._parameters[key]), gain=1)
        print('done')
#############
#model = ResNet_TO_LSTM()
#for i in model.modules():
#    weight_init(i)


#各类别错误个数
def accuracy_of_label(true_list, pred_list):
    A_error = 0
    F_error = 0
    G_error = 0
    K_error = 0
    for i in range(len(true_list)):
        if true_list[i] == 0 and pred_list[i] != 0:
            A_error += 1
        if true_list[i] == 1 and pred_list[i] != 1:
            F_error += 1
        if true_list[i] == 2 and pred_list[i] != 2:
            G_error += 1
        if true_list[i] == 3 and pred_list[i] != 3:
            K_error += 1
    return A_error, F_error, G_error, K_error

#####################################################################################################################
##加载数据
def Load_data(str):
    if str == 'train':
        data_index = pd.read_csv('train.csv')
    if str =='valid':
        data_index = pd.read_csv('valid.csv')
    if str =='test':
        data_index = pd.read_csv('test.csv')
    if str == 'debug':
        data_index = pd.read_csv('debug.csv')
    #data_index = data_index.to_numpy().astype(np.int)
    if ALL_GPU == True:
        dataset = spectrum_data_GPU(data_index,data_mapping,data)
    else:
        dataset = spectrum_data(data_index)
    return dataset
####################################################################################################################
#预加载数据
class data_prefetcher():
    def __init__(self, loader, test=False):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.test_flag = test
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            if self.test_flag == True:
                self.next_input = self.next_data[:,:-1].cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)
                self.next_name = self.next_data[:,-1].cuda(non_blocking=True)
            else:
                self.next_input = self.next_data.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.reshape(-1,1,3738).float()
            #self.next_input = self.next_input.reshape(-1, 42, 89).float()
            self.next_target = self.next_target.float()#long用于分类。float用于预测
            if self.test_flag:
                self.next_name = self.next_name.to('cpu').numpy().reshape(1,-1)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if self.test_flag:
            name = self.next_name
        self.preload()
        if self.test_flag == True:
            return input, target, name
        else:
            return input, target

def error_name(p, data_name, correct):
    with open('error_name.txt', 'a') as f:  # 打开.txt   如果文件不存在，创建该文件。
        for i in range(len(correct)):
            if correct[i] == 0:
                f.write('\n'+str(data_name[i]))  # 把变量v写入.txt。这里var必须是str格式，如果不是，则可以转一下。
                f.write('A:%f\tF:%f\tG:%f\tK:%f'%(p[i][0],p[i][1],p[i][2],p[i][3])) #写入各类别预测概率
    f.close()

def MAE(epoch, pred, label, plt_flag=False, data_name=None):
    abs_error = abs(torch.sub(pred, label))
    print(abs_error.shape)
    max_error = abs_error.max(1)[0]
    MAE_error = (abs_error.sum() - max_error) / (abs_error.shape[1] - 1)
    '''
    all_in = 0
    all_out = 0
    A_in = 0
    A_out = 0
    F_in = 0
    F_out = 0
    G_in = 0
    G_out = 0
    K_in = 0
    K_out = 0
    for i in range(abs_error.shape[1]):
        if abs_error[0][i] > 0.2:
            if  0.0 <=label[0][i] < 1.0:
                A_out += 1
                all_out +=1
            if  1.0 <=label[0][i] < 2.0:
                F_out += 1
                all_out +=1
            if  2.0 <=label[0][i] < 3.0:
                G_out += 1
                all_out +=1
            if  3.0 <=label[0][i] < 4.0:
                K_out += 1
                all_out +=1
        if abs_error[0][i] <= 0.2:
            if  0.0 <=label[0][i] < 1.0:
                A_in += 1
                all_in +=1
            if  1.0 <=label[0][i] < 2.0:
                F_in += 1
                all_in +=1
            if  2.0 <=label[0][i] < 3.0:
                G_in += 1
                all_in +=1
            if  3.0 <=label[0][i] < 4.0:
                K_in += 1
                all_in +=1
    with open('error_num.txt','a') as f:
        f.write('all_in:%d\nall_out:%d\nA_in:%d\nA_out:%d\nF_in:%d\nF_out:%d\nG_in:%d\nG_out:%d\nK_in:%d\nK_out:%d\n'%(all_in,all_out,A_in,A_out,F_in,F_out,G_in,G_out,K_in,K_out))
'''
    if plt_flag:
        sns.distplot(abs_error.cpu().detach().numpy(), bins=40, hist=True, norm_hist=False, rug=True, vertical=False, axlabel='Error', kde=False)
        plt.ylabel('Total Number')
        plt.title('Error Distribution')
        plt.tight_layout()
        plt.savefig('error_distribution')
        plt.clf()
        error_name_MAE(epoch, data_name, pred, label, abs_error)
    std_error = np.std(abs_error.cpu().detach().numpy())   
    return MAE_error.item(), max_error.item(), std_error

def error_name_MAE(epoch, data_name, pred, label, abs_error):
    val, indices = abs_error.sort(descending=True)
    with open('error_name.txt', 'a') as f:
        f.write('\n'+'############第%d次写入###########'%epoch)
        for i in range(val.shape[1]):
            if val[0][i] >= 0.5:
               f.write('\n'+ str(data_name[0][indices[0][i]].item()))
               f.write('\tlabel:%.2f\tpred:%.2f\t'%(label[0][indices[0][i]],pred[0][indices[0][i]]))
    f.close()
        

def pred_label_plot(pred, label):
    sns.set_style('whitegrid')
    sns.jointplot(pred.squeeze().cpu().detach().numpy(), label.squeeze().cpu().detach().numpy(),scatter_kws={"s": 0.05}, kind="reg",space=.5, xlim=(0,4), ylim=(0,4),joint_kws={'y_jitter':.05, 'order':2,'ci':95})
    plt.title('Linear Regression')
    plt.xlabel('Pred')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig('pred_label')
    plt.clf()

class logcosh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        loss = torch.log(torch.cosh(x - y))
        return torch.sum(loss)
