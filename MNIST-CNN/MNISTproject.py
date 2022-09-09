import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#(디스플레이가 있는 환경만 가능)
# from matplotlib import pyplot as plt
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

# MNIST 데이터 확인 (디스플레이가 있는 환경만 가능)
# image, label = train_data[0]
# plt.imshow(image.squeeze().numpy(), cmap = 'gray')
# plt.title('label : %s' % label)
# plt.show()

#미니 배치 구성
train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          batch_size = batch_size,
                                          shuffle = True)
first_batch = train_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))


# CNN Structural Design
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128, 10)
    

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output

    
#Adam 알고리즘의 optimizer 지정 > 손실함수를 최소로 하는 가중치를 찾기위해  
#다중 클래스 분류 문제이기에 교차 엔트로파를 손실함수로 지정
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
criterion = nn.CrossEntropyLoss()

print(model)
# CNN(
#   (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
#   (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
#   (dropout1): Dropout2d(p=0.25, inplace=False)
#   (dropout2): Dropout2d(p=0.5, inplace=False)
#   (fc1): Linear(in_features=9216, out_features=128, bias=True)
#   (fc2): Linear(in_features=128, out_features=10, bias=True)
# )
model.train()
i = 0 
for epoch in range(epoch_num):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('Train Step: {}\tLoss: {:.3f}'.format(i, loss.item()))
        i += 1

model.eval()
correct = 0 
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('test set: Accuracy: {:.2f}%'.format((100 * correct / len(test_loader.dataset))))