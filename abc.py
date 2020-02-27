import torch
import matplotlib.pyplot as plt
import seaborn as sns
from function import *
import MODEL
from torchvision import transforms, models, datasets

a = np.array([1,2,3],dtype=np.float)
b = np.array([5,5,5],dtype=np.float)
print(a)
print(b)
c = [a,b]

np.concatenate(c,axis=0)
