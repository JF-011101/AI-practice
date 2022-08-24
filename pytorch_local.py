import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
import sys,getopt

# 卷积层使用 torch.nn.Conv2d
# 激活层使用 torch.nn.ReLU
# 池化层使用 torch.nn.MaxPool2d
# 全连接层使用 torch.nn.Linear
 
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(3,6,3,1,2),nn.ReLU(),nn.MaxPool2d(2, 2))
    self.conv2 = nn.Sequential(nn.Conv2d(6,16,5),nn.ReLU(),nn.MaxPool2d(2, 2))
    self.fc1 = nn.Sequential(nn.Linear(16*5*5,120),nn.BatchNorm1d(120),nn.ReLU())
    self.fc2 = nn.Sequential(nn.Linear(120,84),nn.BatchNorm1d(84),nn.ReLU(),nn.Linear(84, 6))
 
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size()[0],-1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
  epochs = 100
  train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(root="./flower_photos",transform=transforms.Compose([transforms.RandomResizedCrop(28),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),batch_size=64,shuffle=True)
  net = CNN().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(),lr=0.001)
  train_num = len(train_loader)
  for epoch in range(epochs):
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
      inputs, labels = Variable(data[0]).cuda(), Variable(data[1]).cuda()
      optimizer.zero_grad()
      loss = criterion(net(inputs), labels)
      loss.backward()
      optimizer.step()
      sum_loss += loss.item()
    print('[epoch %d] train_loss: %.3f' %(epoch + 1, sum_loss / train_num))
  torch.save(net.state_dict(), './Net.pth')

def predict():
  data_transform = transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  for nums in range(1, 7):
    img = torch.unsqueeze(data_transform(Image.open('./TestImages/test' + str(nums) + '.jpg')), dim=0)
    with open('./class_indices.json', "r") as f:
      class_indict = json.load(f)
    net = CNN().to(device)
    net.load_state_dict(torch.load("./Net.pth"))
    net.eval()
    with torch.no_grad():
      predict = torch.softmax(torch.squeeze(net(img.to(device))).cpu(), dim=0)
      predict_cla = torch.argmax(predict).numpy()
    print('test'+ str(nums) +"\t\t\tpredict result: {}\t\t\tprobability: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy()))

if __name__ == '__main__':
  if 'train' in sys.argv:
    train()
  elif 'predict' in sys.argv:
    predict()
