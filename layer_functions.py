# define special layer in this file

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Function

from util_functions import *

"""
-----------  functions that support positize ----------
"""

# backwardについて工夫をしていない実装
class PositizeFunction(Function):
  @staticmethod
  def forward(ctx,x,bit_size,es_size):
    y = floatTensor_to_positTensor(x,bit_size,es_size)
    return y

  @staticmethod
  def backward(ctx, grad_output):
    # inputと同じ数だけ返せば良い
    return grad_output, None, None



# backwardについて工夫をした実装
class PositizeFunction2(Function):
  @staticmethod
  def forward(ctx,x,bit_size,es_size):
    y = floatTensor_to_positTensor(x,bit_size,es_size)
    saved = torch.div(y,x)
    ctx.bit_size = bit_size
    ctx.es_size = es_size
    ctx.save_for_backward(saved)
    return y

  @staticmethod
  def backward(ctx, grad_output):
    saved, = ctx.saved_tensors
    # saved 0が含まれている可能性がある(0/0)ので，それを1に直す作業を行う
    # floatのequality checkは難しいので，positのminいかになっていれば1へと変換するようにする
    bit_size = ctx.bit_size
    es_size = ctx.es_size
    useed = 2 ** (2 ** es_size)
    minpos = useed ** (2 - bit_size)
    maxpos = useed ** (bit_size - 2)
    min_judge_tensor = torch.full_like(saved, minpos/2)
    saved = torch.where(torch.isnan(saved),torch.full_like(saved,1),saved)
    saved = torch.where(abs(saved)<minpos,torch.full_like(saved,1),saved)
    saved = torch.where(abs(saved)>maxpos,torch.full_like(saved,1),saved)
    # print(torch.max(abs(saved),1))
    # TODO: 応急処置的にやった処理なので改良が必要
    return grad_output*(saved), None, None

def positize(x,bit_size,es_size):
  return PositizeFunction.apply(x,bit_size,es_size)





"""
---------- Definition of Positized Layers ----------
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
  def __init__(self, input_feature, output_feature, bit_size, es_size):
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
  def __init__(self, input_channel, output_channel, kernel_size, bit_size, es_size, stride=1, padding=0):
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