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

class MyTrainData(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x = x_train
        self.y = y_train
        if len(x_train) == len(y_train):
            self.len = len(x_train)
        else:
            print("Error: x_train y_train have different lengths")