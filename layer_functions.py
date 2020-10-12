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
    return floatTensor_to_positTensor(x,bit_size,es_size)


  @staticmethod
  def backward(ctx, grad_output):
    # inputと同じ数だけ返せば良い
    return grad_output, None, None

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


