import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from data_functions import *

# import various neural networks
from dnn import *
from util_functions import *



def main():

  # set hyper parameters for Posit-NN
  bit_size = 8
  es = 1

  # batch size for train and test
  batch_train = 100
  batch_test = 1000

  # load data
  if(sys.argv[1]=='mnist'):
    dataset_train, dataset_test = generate_datasets('mnist')
    dataloader_train, dataloader_test = generate_dataloader(dataset_train, dataset_test,batch_train=batch_train, batch_test=batch_test)


  # define NN
  if(sys.argv[2] == 'EasyNN'):
    net = EasyNN()
  elif(sys.argv[2] == 'EasyPNN'):
    net = EasyPNN()

  ## define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)


  # variables for visualize data
  epochs_for_visualize = np.array([])
  loss_for_visualize = np.array([])
  accuracy_for_visualize = np.array([])


  # train
  epochs = 2
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
      if i % batch_train == batch_train-1:
        print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch+1,i+1,running_loss/batch_train))
        remembered_running_loss = running_loss/batch_train
        running_loss = 0.0

    # remember accuracy for each epochs
    loss_for_visualize = np.append(loss_for_visualize,remembered_running_loss)
    accuracy_for_visualize = np.append(accuracy_for_visualize, calculate_accuracy(dataloader_test,net))
    epochs_for_visualize = np.append(epochs_for_visualize, epoch+1)


  print('Training Finished')

  # test
  print('Accuracy: {:2f} %%'.format(calculate_accuracy(dataloader_test,net)))



  # visualize
  plt.figure()
  plt.title(sys.argv[1]+' '+sys.argv[2]+' acc')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.ylim(0,100)
  plt.plot(epochs_for_visualize,accuracy_for_visualize)
  plt.savefig(sys.argv[1]+' '+sys.argv[2]+' acc.png')


  plt.figure()
  plt.title(sys.argv[1]+' '+sys.argv[2]+' loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.plot(epochs_for_visualize,loss_for_visualize)
  plt.savefig(sys.argv[1]+' '+sys.argv[2]+' loss.png')


if __name__ == "__main__":
  main()