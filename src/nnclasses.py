# Load libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# Define neural network
class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 10)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Define DataSet subclass for training data
class MyTrainData(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x = x_train
        self.y = y_train
        if len(x_train) == len(y_train):
            self.len = len(x_train)
        else:
            print("Error: x_train y_train have different lengths")
    def __getitem__(self, i):
        z = self.x[i], self.y[i]
        return z

    def __len__(self):
        return self.len

# Define DataSet class for test data
class MyTestData(Dataset):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x = x_test
        self.y = y_test
        if len(x_test) == len(y_test):
            self.len = len(x_test)
        else:
            print("Error: x_test and y_test have different lengths")
    def __getitem__(self, i):
        z = self.x[i], self.y[i]
        return z

    def __len__(self):
        return self.len