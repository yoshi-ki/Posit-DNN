import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from data_functions import *

# import various neural networks
from dnn import *



def main():

  # set hyper parameters for Posit-NN
  bit_size = 8
  es = 1

  # load data
  dataset_train, dataset_test = generate_datasets("mnist")
  dataloader_train, dataloader_test = generate_dataloader(dataset_train, dataset_test,batch_train=100, batch_test=1000)


  # define NN
  net = EasyNN()

  ## define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)


  # train
  epochs = 5
  for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader_train):
      optimizer.zero_grad()

      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step() # update parameters

      # print statistics
      running_loss += loss.item()
      if i % 100 == 99:
        print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch+1,i+1,running_loss/100))
        running_loss = 0.0
  print('Training Finished')

  # test
  correct = 0
  total = 0
  with torch.no_grad():
    for(images, labels) in dataloader_test:
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted==labels).sum().item()
  print('Accuracy: {:2f} %%'.format(100 * float(correct/total)))



  # visualize





if __name__ == "__main__":
  main()