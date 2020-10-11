import numpy as np
import torch
import torch.nn as nn


from data_functions import *



def main():

  # set hyper parameters for Posit-NN
  bit_size = 8
  es = 1

  # load data
  dataset_train, dataset_test = generate_datasets("mnist")
  dataloader_train, dataloader_test = generate_dataloader(dataset_train, dataset_test,batch_train=100, batch_test=1000)


  # define NN



  # train


  # visualize





if __name__ == "__main__":
  main()