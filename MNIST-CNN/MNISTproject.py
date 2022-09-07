import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
# %maplotlib inline #주피터 사용시에 사용 > 이미지 바로 확인 가능 

#setting
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('Current cuda device is', device)

print(torch.cuda.get_device_name()) # GPU 이름 

print(torch.cuda.device_count()) # 사용 가능 GPU 개수 

#
batch_size = 50
epoch_num = 15
learning_rate = 0.0001

#데이터 불러오기 
train_data = datasets.MNIST(root = './data', train = True, download = True,
                           transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train = False,
                           transform = transforms.ToTensor())
print('트레이닝 데이터 개수: ', len(train_data))
print('테스트 데이터 개수: ', len(test_data))
