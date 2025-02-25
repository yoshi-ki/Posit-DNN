# define Neural Network in this file

import numpy as np
import torch
import torch.nn as nn

from layer_functions import *


# EasyNN is for test
class EasyNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
    self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64
    self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
    self.dropout1 = nn.Dropout2d()
    self.fc1 = nn.Linear(12 * 12 * 64, 128)
    self.dropout2 = nn.Dropout2d()
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = nn.functional.relu(x)
    x = self.conv2(x)
    x = nn.functional.relu(x)
    x = self.pool(x)
    x = self.dropout1(x)
    x = x.view(-1, 12 * 12 * 64)
    x = nn.functional.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.fc2(x)
    return x



# EasyPNN is for test
class EasyPNN(nn.Module):
  def __init__(self,bit_size,es_size):
    super().__init__()
    self.bit_size = bit_size
    self.es_size = es_size
    self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
    self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64
    self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
    self.dropout1 = nn.Dropout2d()
    self.fc1 = nn.Linear(12 * 12 * 64, 128)
    self.dropout2 = nn.Dropout2d()
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    bit_size = self.bit_size
    es_size = self.es_size
    x = positize(x,bit_size,es_size)
    x = self.conv1(x)

    x = positize(x,bit_size,es_size)
    x = nn.functional.relu(x)

    x = self.conv2(x)

    x = positize(x,bit_size,es_size)
    x = nn.functional.relu(x)

    x = self.pool(x)
    x = positize(x,bit_size,es_size)

    x = self.dropout1(x)
    x = x.view(-1, 12 * 12 * 64)

    x = nn.functional.relu(self.fc1(x))

    x = self.dropout2(x)

    x = self.fc2(x)
    x = positize(x,bit_size,es_size)
    return x



# EasyPNN2 is for test
class EasyPNN2(nn.Module):
  def __init__(self,bit_size,es_size):
    super().__init__()
    self.conv1 = PositizedConv2d(1, 32, 3,bit_size,es_size) # 28x28x32 -> 26x26x32
    self.conv2 = PositizedConv2d(32, 64, 3,bit_size,es_size) # 26x26x64 -> 24x24x64
    self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
    self.dropout1 = nn.Dropout2d()
    self.fc1 = PositizedLinear(12 * 12 * 64, 128,bit_size,es_size)
    self.dropout2 = nn.Dropout2d()
    self.fc2 = PositizedLinear(128, 10,bit_size,es_size)

  def forward(self, x):
    x = self.conv1(x)
    x = nn.functional.relu(x)
    x = self.conv2(x)
    x = nn.functional.relu(x)
    x = self.pool(x)
    x = self.dropout1(x)
    x = x.view(-1, 12 * 12 * 64)
    x = nn.functional.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.fc2(x)
    return x




class VGG11(torch.nn.Module):

  def __init__(self):
    super().__init__()

    self.block0 = nn.Sequential(
      nn.Conv2d(3,64,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block1 = nn.Sequential(
      nn.Conv2d(64,128,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block2 = nn.Sequential(
      nn.Conv2d(128,256,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(256,256,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block3 = nn.Sequential(
      nn.Conv2d(256,512,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block4 = nn.Sequential(
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(512,512,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block5 = nn.Sequential(
      nn.Linear(512,4096),
      nn.ReLU(),
      nn.Linear(4096,4096),
      nn.ReLU(),
      nn.Linear(4096,10)
    )

  def forward(self,x):
    x = self.block0(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = x.view(x.size(0), -1)
    x = self.block5(x)
    return x



class Positized_VGG11(torch.nn.Module):

  def __init__(self, bit_size, es_size):
    super().__init__()

    self.block0 = nn.Sequential(
      PositizedConv2d(3,64,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block1 = nn.Sequential(
      PositizedConv2d(64,128,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block2 = nn.Sequential(
      PositizedConv2d(128,256,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      PositizedConv2d(256,256,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block3 = nn.Sequential(
      PositizedConv2d(256,512,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      PositizedConv2d(512,512,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block4 = nn.Sequential(
      PositizedConv2d(512,512,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      PositizedConv2d(512,512,kernel_size=3,padding=1,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.block5 = nn.Sequential(
      PositizedLinear(512,4096,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      PositizedLinear(4096,4096,bit_size=bit_size,es_size=es_size),
      nn.ReLU(),
      PositizedLinear(4096,10,bit_size=bit_size,es_size=es_size)
    )

  def forward(self,x):
    x = self.block0(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = x.view(x.size(0), -1)
    x = self.block5(x)
    return x


