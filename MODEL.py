from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
################################################################################################################
##ResNet

class GeLU(nn.Module):

    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, input):
        return F.gelu(input)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=5, stride=stride, padding=2, bias=False),
            #nn.BatchNorm1d(outchannel),
            #nn.ReLU(inplace=True),
            GeLU(),
            nn.Conv1d(outchannel, outchannel, kernel_size=5, stride=1, padding=2, bias=False),
            #nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
              # nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

##################################################################################################################
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=4):
        super(ResNet, self).__init__()
        self.inchannel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 32,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer4_1 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.fc_1 = nn.Linear(256, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool1d(self.layer1(out),3)
        out = F.max_pool1d(self.layer2(out),2)
        out = F.max_pool1d(self.layer3(out),2)
        out = F.max_pool1d(self.layer4_1(out),3)
        out = F.avg_pool1d(out,13)
        out = out.view(out.size(0), -1)
        #out = F.dropout(out, p=0.5)
        out = self.fc_1(out)


        return F.log_softmax(out, dim=1)


def ResNet_18():

    return ResNet(ResidualBlock)
################################################################################################################
##LSTM
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, dropout=0, bias=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        #out = F.dropout(out, p=0.5)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out
############################################################################################################
#ConvLSTM
class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = (1, kernel_size)
        #self.num_features = 4

        self.padding = (0, int((kernel_size - 1) / 2))

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):  # 定义一个多层的convLSTM（即多个convLSTMCell），并存放在_all_layers列表中
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):  # 在每一个时步进行前向运算
            x = input
            for i in range(self.num_layers):  # 对多层convLSTM中的每一层convLSTMCell，依次进行前向运算
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:  # 如果是在第一个时步，则需要调用init_hidden进行convLSTMCell的初始化
                    print(x.size())
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)  # 调用convLSTMCell的forward进行前向运算
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


#convlstm = ConvLSTM(input_channels=4, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=4,effective_step=[3])

#######################################################################################################################
##CNN + LSTM

class CNN_TO_LSTM(nn.Module):
    def __init__(self, ResidualBlock, in_dim = 32, hidden_dim = 32, n_layer = 5, num_classes=4):
        super(CNN_TO_LSTM, self).__init__()
        self.inchannel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 32,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, dropout=0, bias=True)
        self.fc1 = nn.Linear(hidden_dim,num_classes)
        self.fc_ = nn.Linear(1344, num_classes)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out_list = []
        for i in range(x.shape[1]):
            out = self.conv1(x[:,i,:].view(x.size(0),1,-1))
            out = self.layer1(out)
            out = F.max_pool1d(self.layer2(out),2)
            out = F.max_pool1d(self.layer3(out),2)
            out = F.max_pool1d(self.layer4(out),3)
            out_list.append(out)
        out = reduce(lambda a,b:torch.cat((a,b),dim=2),out_list)
        print(out.shape)
        out = out.permute(0,2,1)
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        #out = out.view(out.size(0),-1)
        #out = self.fc_(out)
        out = self.fc1(out)
        return F.log_softmax(out, dim=1)

def ResNet_TO_LSTM():

    return CNN_TO_LSTM(ResidualBlock)


class DeepCNN(nn.Module):
    def __init__(self, ResidualBlock, num_classes=1):
        super(DeepCNN, self).__init__()
        self.inchannel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=0, bias=False),
            #nn.BatchNorm1d(32),
            #nn.ReLU(),
            GeLU(),
        )
        self.maxpool1d = nn.MaxPool1d(kernel_size = 3)
        self.layer1 = self.make_layer(ResidualBlock, 32,  2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 64,  2, stride=3)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2, stride=3)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=3)
        self.layer5 = self.make_layer(ResidualBlock, 512, 2, stride=3)
        self.layer6 = self.make_layer(ResidualBlock, 1024, 2, stride=3)
        #self.outchannel = nn.Conv1d(256,4,kernel_size=1,stride=1,padding=0,bias=False)
        self.avgpool1d = nn.AvgPool1d(2)  #修改为2
        
        self.fc_regression = nn.Linear(1024, num_classes)
        #self.fc_classifiction = nn.Linear(1024, 4)
        #self.fc_classifiction = nn.Linear(4, 4)#  删除
        #self.softmax = nn.LogSoftmax(dim=1)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1d(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #out = self.outchannel(out)
        out = self.avgpool1d(out)
        out = out.view(out.size(0), -1)
        out = self.fc_regression(out)
        #out = self.fc_classifiction(out)
        #out = self.softmax(out)
        return out

def Deep_CNN():

    return DeepCNN(ResidualBlock)

#input = torch.rand(2, 1, 3738)
#input = torch.rand(2, 42, 89)

#import numpy as np
#model = Deep_CNN()
#out = model(input)
#print(out.shape)
#params = list(model.cpu().parameters())
#print(len(params))
#weight_softmax = np.squeeze(params[-2].data.numpy())
#print(weight_softmax.shape)    

#print(out)
#print(F.softmax(out))
#for i in model.modules():
#    print(i)
