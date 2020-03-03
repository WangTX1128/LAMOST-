import torch
import matplotlib.pyplot as plt
import seaborn as sns
from function import *
import MODEL
from torchvision import transforms, models, datasets

a = torch.tensor([[1,2,3],[4,5,6]])
a = a.sum(axis=1)
print(torch.min(a))