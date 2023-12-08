# Import libraries

import numpy as np
import pandas as pd
import random
import os
import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from plotnine import ggplot, aes, geom_point

random.seed(123)
torch.manual_seed(456)

x_vec = np.array(np.random.rand(100, 1)).astype("float32")
def quad(x):
    return(0.5*x*x + x)

y_vec = np.array(list(map(quad,  x_vec))).astype("float32")

x_train = torch.tensor(x_vec, requires_grad = True)
y_train = torch.tensor(y_vec, requires_grad=True)

df_imp = pd.DataFrame({"x":x_vec.reshape([-1, ]), "y":y_vec.reshape([-1, ])})

# Plot imput data
ggplot(df_imp, aes(x = "x", y="y")) + geom_point(color="blue")

# Create dataset
class MyDataSet(Dataset):
    def __init__(self, len):
        super().__init__()
        self.x = x_train
        self.y = y_train
        self.len = len
    def __getitem__(self, i):
        z = self.x[i], self.y[i]
        return z
    def __len__(self):
        return self.len

MyData = MyDataSet(100)
dataloader = DataLoader(MyData, batch_size=1)

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

model = MyNN()

print(list(model.parameters()))


# Start training loop
EPOCHS = 100

list_loss = []
loss_function = nn.MSELoss()
sum_loss = 0
optim = torch.optim.SGD(model.parameters(), lr=1e-1)

# Train neural network
for epoch in range(EPOCHS):
    for u in dataloader:
        output = model(MyData.x)
        loss = loss_function(output, MyData.y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        sum_loss += loss
        list_loss.append(round(float(loss.detach().numpy()), 4))
        if epoch % 10 == 0:
            print("========Epoch: {}".format(str(epoch)))
            print(round(float(loss.detach().numpy())), 4)

output_vec = model(MyData.x).detach().numpy()
df_model = pd.DataFrame({"x":x_vec.reshape([-1, ]),
"y":output_vec.reshape([-1, ])})

# Plot model output
ggplot(df_model, aes(x="x", y="y")) + geom_point()

# Concatenate data
concatenated_x = np.concatenate((x_vec.reshape([-1, ]),
x_vec.reshape([-1,])))
concatenated_y = np.concatenate((output_vec.reshape([-1, ]),
y_vec.reshape([-1, ])))
flag_y = int(len(concatenated_x)/2)*["ground_truth_y"] + int(len(concatenated_x)/2)*["model_output"]

df_consolidate = pd.DataFrame({"x":concatenated_x, "y":concatenated_y, "flag":flag_y})

# Plot model output
ggplot(df_consolidate, aes(x="x", y="y", color="flag")) + geom_point()
