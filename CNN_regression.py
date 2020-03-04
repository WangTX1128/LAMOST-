import torch
from MODEL import *
from function import *
from torch import optim
from tensorboardX import SummaryWriter
from training_acceleration import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

writer = SummaryWriter('runs/localCNN')

def train_net(epoch, model, data_trainer, criterion, optimizer):
    running_loss = 0.0
    correct = 0
    train_loss = 0.0
    pred_list = torch.zeros(1, 1, dtype=torch.float).to('cuda')  # 网络预测值
    true_list = torch.zeros(1, 1, dtype=torch.float).to('cuda')  # 真实值
    model.train()
    prefetcher = data_prefetcher(data_trainer, test=False)
    data, label = prefetcher.next()
    batch_idx = 0
    while data is not None:
        batch_idx += 1
        '''
        if torch.any(torch.isnan(data)) | torch.any(torch.isnan(label)):
            print('Find nan in datas or labels')
            raise TypeError
        '''
        optimizer.zero_grad()
        output = model(data)
        pred_list = torch.cat((pred_list, output.view(1, -1)), 1)
        true_list = torch.cat((true_list, label.view(1, -1)), 1)
        loss = criterion(output.squeeze(1), label)
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
    #true_list = true_list.squeeze().long().to('cpu').tolist()
    #pred_list = pred_list.squeeze().long().to('cpu').tolist()
    MAE_error, max_error, std_error = MAE(epoch, pred_list[:,1:], true_list[:,1:], plt_flag=False)
    #A_error, F_error, G_error, K_error = accuracy_of_label(true_list, pred_list)
    #print(A_error, F_error, G_error, K_error)
    print('\nTrain set: Average loss: {:.4f}, MAE_error / max_error: {:.2f}/{:.2f}, Std:{:.2f}\n'.format(
        train_loss / len(data_trainer.dataset), MAE_error, max_error, std_error))

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
    
    writer.add_scalar('valid loss', test_loss, epoch)
    
    MAE_error, max_error, std_error = MAE(epoch, pred_list[:,1:], true_list[:,1:], plt_flag=True, data_name=name_array[:,1:])
    pred_label_plot(pred_list[:,1:], true_list[:,1:])
    print('\nTest set: Average loss: {:.4f}, MAE_error / max_error: {:.2f}/{:.2f}, Std:{:.2f}\n'.format(
        test_loss, MAE_error, max_error, std_error))
    #true_list = true_list.squeeze().long().to('cpu').tolist()
    #pred_list = pred_list.squeeze().long().to('cpu').tolist()
    #F1_score = f1_score(true_list, pred_list, average='macro')

    #writer.add_scalar('valid F1 Score',
    #                  F1_score,
    #                  epoch )

    #A_error, F_error, G_error, K_error = accuracy_of_label(true_list, pred_list)
    #print(A_error, F_error, G_error, K_error)
    valid_MAE_error_list.append(MAE_error)
    #print('\nvalid_F1_score = {:.4f}\n'.format(F1_score))




if __name__ == '__main__':
    learning_rate = 0.0003
    epoch = 2000
    torch.backends.cudnn.benchmark = True
    Frozen_Layers = False
    train_set = load_data_GPU('train')
    valid_set = load_data_GPU('valid')

    valid_MAE_error_list = []
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    #model = ResNet_18()
    #model = ResNet_TO_LSTM()
    model = Deep_CNN() #回归模型
    for i in model.modules():
        weight_init(i)
    device = torch.device('cuda')
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 3.774, 14.49, 1]))
    criterion = logcosh()
    criterion.to(device)### 要想使用GPU，此处必须使用cuda()

    if Frozen_Layers == True:
        for k, v in model.named_parameters():
            if 'lstm' not in k and 'fc' not in k:
                v.requires_grad = False
        print('CNN FROZENED!')
        for k, v in model.named_parameters():
            if v.requires_grad == True:
                print('training:',k)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0)

    bias_list = (param for name, param in model.named_parameters() if 'bias' in name and param.requires_grad == True)
    others_list = (param for name, param in model.named_parameters() if 'bias' not in name and param.requires_grad == True)
    parameters = [{'params': bias_list, 'weight_decay': 0},
                  {'params': others_list}]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    model.to(device)

    #加载LSTM预训练参数
    #pretrained_params = torch.load('best_model.pkl')
    #model.load_state_dict(pretrained_params, strict=False)
    model.load_state_dict(torch.load('deep_cnn_regression_cosh.pkl'),strict=False)
    
    mae_err = 9999
    
    for i in range(1, epoch+1):

        train_net(i, model, train_loader, criterion, optimizer)

        test_net(i, model, valid_loader, test=True)
        if valid_MAE_error_list[-1] < mae_err:
            mae_err = valid_MAE_error_list[-1]
            torch.save(model.state_dict(), 'deep_cnn_regression_cosh.pkl')  # save only the parameters

    model.load_state_dict(torch.load('deep_cnn_regression_cosh.pkl'))
    test_set = load_data_GPU('test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_net(1, model,test_loader, test=True)