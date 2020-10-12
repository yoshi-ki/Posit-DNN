# functions

import numpy as np
import torch

def float_to_posit(x, bit_size, es_size):

  """
  input : x is treated as real number
  output : (bit_size, es) posit
  """

  # set parameters for posit
  n = bit_size
  es = es_size

  # algorithm for transration
  useed = 2 ** (2 ** es)
  maxpos = useed ** (n - 2)
  minpos = useed ** (2 - n)
  threshold = minpos / 2
  if (abs(x) < threshold):
    return 0
  else:
    s = np.sign(x)
    x_hat = max(minpos, min(abs(x), maxpos))
    E = np.floor(np.log2(x_hat))
    r = np.floor(E / (2 ** es))
    e = np.mod(E, 2 ** es)
    m = x_hat / (2 ** E) - 1
    b_r = 0 # regimeのbit数
    if r >= 0:
      b_r = r + 2
    else:
      b_r = -r + 1
    b_e = min(n - 1 - b_r, es)
    b_m = max(n - 1 - b_r - b_e, 0) # mantissaのbit数
    e_p = np.floor(e * (2 ** (b_e - es))) * (2 ** (es - b_e))
    m_p = np.floor(m * (2 ** b_m)) * (2 ** (-b_m))
    x_p = s * (useed ** r) * (2 ** e_p) * (1 + m_p)
    return x_p


def floatTensor_to_positTensor(x, bit_size, es_size):

  """
  input : x is treated as real number tensor
  output : (bit_size, es) posit tensor
  """

  # set parameters for posit
  n = bit_size
  es = es_size

  # algorithm for transration
  useed = 2 ** (2 ** es)
  maxpos = useed ** (n - 2)
  minpos = useed ** (2 - n)
  threshold = minpos / 2
  x = torch.where(abs(x)>=threshold, x, torch.zeros_like(x))
  s = torch.sign(x)
  x = torch.where(abs(x)<=maxpos,x,torch.full_like(x,maxpos))
  x = torch.where(abs(x)>=minpos,x,torch.full_like(x,minpos))
  x_hat = abs(x)
  E = torch.floor(torch.log2(x_hat))
  r = torch.floor(torch.div(E,2**es))
  e = E-(2**es)*r
  m = torch.div(x_hat,(2 ** E)) - 1
  b_r = torch.where(r>=0,r+2,-r+1)
  b_e = torch.where(n-1-b_r<es,n-1-b_r,torch.full_like(n-1-b_r,es)) # esのbit数
  b_m = torch.where(n-1-b_r-b_e>=0,n-1-b_r-b_e,torch.zeros_like(n-1-b_r-b_e)) # mantissaのbit数
  e_p = torch.floor(e * (2 ** (b_e - es))) * (2 ** (es - b_e))
  m_p = torch.floor(m * (2 ** b_m)) * (2 ** (-b_m))
  x_p = s * (useed ** r) * (2 ** e_p) * (1 + m_p)
  return x_p





# calculate accuracy
def calculate_accuracy(dataloader_test,net):
  correct = 0
  total = 0
  with torch.no_grad():
    for(images, labels) in dataloader_test:
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted==labels).sum().item()
  return (100 * float(correct/total))



if __name__ == "__main__":
  #test code for floatTensor_to_positTensor
  a = torch.randn(5)
  print(a, floatTensor_to_positTensor(a,8,1))