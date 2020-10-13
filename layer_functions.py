# define special layer in this file

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Function

from util_functions import *

"""
functions that support positize
"""
class PositizeFunction(Function):
  @staticmethod
  def forward(ctx,x,bit_size,es_size):
    y = floatTensor_to_positTensor(x,bit_size,es_size)
    ctx.save_for_backward(x,y)
    return y


  @staticmethod
  def backward(ctx, grad_output):
    # inputと同じ数だけ返せば良い
    x, y = ctx.saved_tensors
    # return grad_output, None, None
    # TODO: 本当は下が正しいはず
    return grad_output*(torch.div(y, x)), None, None

def positize(x,bit_size,es_size):
  return PositizeFunction.apply(x,bit_size,es_size)





"""
Positized Layers
"""
class Positize(nn.Module):
  def __init__(self,bit_size, es_size):
    super().__init__()
    self.bit_size = bit_size
    self.es_size = es_size
  def forward(self,x):
    y = positize(x,bit_size=self.bit_size,es_size=self.es_size)
    return y




class PositizedLinear(nn.Module):
  def __init__(self, input_feature, output_feature, bit_size=8, es_size=1):
    super().__init__()
    self.linear = nn.Linear(input_feature,output_feature)
    self.bit_size = bit_size
    self.es_size = es_size
  def forward(self,x):
    w = self.linear.weight.data
    b = self.linear.bias.data
    x_positized = positize(x,self.bit_size,self.es_size)
    w_positized = positize(w,self.bit_size,self.es_size)
    b_positized = positize(b,self.bit_size,self.es_size)
    self.linear.weight.data = w_positized
    self.linear.bias.data = b_positized
    y = self.linear(x_positized)
    return y



class PositizedConv2d(nn.Module):
  def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bit_size=8, es_size=1):
    super().__init__()
    self.conv2d = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
    self.bit_size = bit_size
    self.es_size = es_size
  def forward(self,x):
    w = self.conv2d.weight.data
    x_positized = positize(x,self.bit_size,self.es_size)
    w_positized = positize(w,self.bit_size,self.es_size)
    self.conv2d.weight.data = w_positized
    y = self.conv2d(x_positized)
    return y