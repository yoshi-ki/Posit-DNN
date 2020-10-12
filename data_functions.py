#functions for data

import numpy as np

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader



def generate_datasets(name):
  if(name == "mnist"):
    root = "./Datasets"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    dataset_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
  return dataset_train, dataset_test


def generate_dataloader(dataset_train, dataset_test, batch_train=1, batch_test=1):
  dataloader_train = DataLoader(dataset_train, batch_size=batch_train, shuffle=True)
  dataloader_test = DataLoader(dataset_test, batch_size=batch_test, shuffle=False)
  return dataloader_train, dataloader_test