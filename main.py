import sys
import os
import csv
import argparse

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


  """

  ---------- Parser ----------


  """

  # parser
  parser = argparse.ArgumentParser(description='neural network framework for posit')
  parser.add_argument('dataset', help = 'specify datasets')
  parser.add_argument('model', help = 'specify models')
  parser.add_argument('--bit_size', type=int, default=16, help = 'specify bit size for posit (default is 16)')
  parser.add_argument('--es_size', type=int, default=1, help = 'specify es size for posit (default is 1')
  parser.add_argument('--batch_train', type=int, default=100, help='specify batch size of training (default is 100)')
  parser.add_argument('--batch_test', type=int, default=1000, help='specify batch size of testing (default is 1000)')
  parser.add_argument('--epochs', type=int, default=5, help='specify epochs (default is 5)')
  args = parser.parse_args()

  # set hyper parameters for Posit-NN
  bit_size = args.bit_size
  es_size = args.es_size

  # batch size for train and test
  batch_train = args.batch_train
  batch_test = args.batch_test

  # set epochs
  epochs = args.epochs


  # load data
  if(args.dataset=='mnist'):
    dataset_train, dataset_test = generate_datasets('mnist')
    dataloader_train, dataloader_test = generate_dataloader(dataset_train, dataset_test,batch_train=batch_train, batch_test=batch_test)
  elif(args.dataset=='cifar10'):
    dataset_train, dataset_test = generate_datasets('cifar10')
    dataloader_train, dataloader_test = generate_dataloader(dataset_train, dataset_test,batch_train=batch_train, batch_test=batch_test)


  # define NN
  if(args.model == 'EasyNN'):
    bit_size=0
    es_size=0
    net = EasyNN()
  elif(args.model == 'EasyPNN'):
    net = EasyPNN(bit_size,es_size)
  elif(args.model == 'EasyPNN2'):
    net = EasyPNN2(bit_size,es_size)
  elif(args.model == 'VGG11'):
    bit_size=0
    es_size=0
    net = VGG11()
  elif(args.model == 'Positized_VGG11'):
    net = Positized_VGG11(bit_size,es_size)

  ## define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  # optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
  optimizer = optim.Adam(net.parameters(), lr=0.0001)


  # variables for visualize data
  epochs_for_visualize = np.array([])
  loss_for_visualize = np.array([])
  accuracy_for_visualize = np.array([])



  """


  ---------- Training ----------


  """


  # train
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
    remembered_accuracy = calculate_accuracy(dataloader_test,net)
    accuracy_for_visualize = np.append(accuracy_for_visualize,remembered_accuracy)
    epochs_for_visualize = np.append(epochs_for_visualize, epoch+1)



    # write record as a csv file for each epoch
    file_path = './Exec_Record/'+ args.dataset+'_'+args.model+ '_' + str(bit_size) + '_' + str(es_size) + '_log.csv'
    with open(file_path, 'a') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerow([epoch+1,remembered_accuracy,remembered_running_loss])


    #10epochごとにmodelの保存をしておく
    if(epoch % 10 == 1):
      torch.save({
        'epoch':epoch,
        'model_state_dict':net.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
      },'./Model_Record/'+args.dataset+'_'+args.model+'_' + str(epochs) + '_' + str(bit_size) + '_' + str(es_size) +'checkpoint.tar')


  # save models after training
  torch.save(net,'./Model_Record/'+args.dataset+'_'+args.model+'_' + str(epochs) + '_' + str(bit_size) + '_' + str(es_size) +'epochs.pth')

  print('Training Finished')




  """


  ---------- Testing ----------


  """

  # test
  print('Accuracy: {:2f} %%'.format(calculate_accuracy(dataloader_test,net)))





  """


  ---------- Visualizing ----------


  """
  # visualize
  plt.figure()
  plt.title(args.dataset+'_'+args.model+'_acc')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.ylim(0,100)
  plt.plot(epochs_for_visualize,accuracy_for_visualize)
  plt.savefig('./Visualize/' + args.dataset+'_'+args.model+'_' + str(bit_size) + '_' + str(es_size) +'_acc.png')


  plt.figure()
  plt.title(args.dataset+'_'+args.model+'_loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.plot(epochs_for_visualize,loss_for_visualize)
  plt.savefig('./Visualize/' + args.dataset+'_'+args.model+'_' + str(bit_size) + '_' + str(es_size) +'_loss.png')


if __name__ == "__main__":
  main()