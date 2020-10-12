# define Neural Network in this file

import numpy as np
import torch
import torch.nn as nn

from layer_functions import *


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
      nn.Linear(4096,1000)
    )

  def forward(self,x):
    x = self.block0(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    return x




