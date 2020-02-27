import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch
from MODEL import Deep_CNN
import matplotlib.pyplot as plt

# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layer

    def forward(self, x):
        for name, module in self.submodule._modules.items():
            if name is "fc_regression":
                x = x.view(x.size(0), -1)
            x = module(x)
            print("moudle_name", name)
            if name in self.extracted_layers:
                return x
            


class VisualSingleFeature():
    def __init__(self, extract_features, save_path):
        self.extract_features = extract_features
        self.save_path = save_path

    def get_single_feature(self):
        print(self.extract_features.shape)  # ex. torch.Size([1, 128, 112, 112])

        extract_feature = self.extract_features[0, :, :]
        print(extract_feature.shape)  # ex. torch.Size([112, 112])

        return extract_feature

    def save_feature_to_img(self):
        # to numpy
        extract_feature = self.get_single_feature().data.numpy()
        # save image
        plt.plot(extract_feature[:,:].T)
        plt.savefig(self.save_path)
        plt.clf()


def single_image_sample():
    img_path = 'D:\PycharmProjects\DataProcess/4类数据集/regression/test_data/'
    input_img = np.loadtxt(img_path,delimiter=',', skiprows=0).astype('float32')[3:].reshape(1,-1)  # 读取图像
    plt.plot(input_img[0])
    plt.savefig('./Deep_CNN/data')
    plt.clf()
    input_tensor = torch.from_numpy(input_img)
    x = input_tensor.unsqueeze(0)
    return x

     # test resnet50 sequential
x = single_image_sample()
print(x.shape)
model = Deep_CNN()
model.load_state_dict(torch.load('deep_cnn_regression_cosh.pkl'))
print(model(x))

for target_sequential in ['conv1', 'maxpool1d','layer1', 'layer2', 'layer3', 'layer4', 'layer5','layer6']:
    myexactor = FeatureExtractor(submodule=model, extracted_layer=target_sequential)
    target_features = myexactor(x)
    savepath = './Deep_CNN/{}'.format(target_sequential)
    VisualSingleFeature(target_features, savepath).save_feature_to_img()
