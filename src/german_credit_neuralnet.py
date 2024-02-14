import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#from nnclasses import MyNN, MyTrainData, MyTestData
#import utils

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


num_features = 27 
hidden_size = 10
learning_rate = 1e-3
num_epochs = 100
print_interval = 10

# Define your neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


input_df_raw = pd.read_csv(r"../data/processed/german_processed.csv")

df_train, df_test = train_test_split(input_df_raw, test_size=0.33, random_state=42)

ytrain = df_train['GoodCustomer']
xtrain = df_train.drop(['GoodCustomer', 'PurposeOfLoan', 'Gender'], axis=1)
ytest = df_test['GoodCustomer']
xtest = df_test.drop(['GoodCustomer', 'PurposeOfLoan', 'Gender'], axis=1)


scaler = StandardScaler()

# Standardize target values to probability

xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# transform target variables to 0 and 1
ytrain = ytrain.map({-1: 0, 1: 1})
ytest = ytest.map({-1: 0, 1: 1})


xtrain = xtrain.astype("float32")
ytrain = ytrain.astype("float32")
xtest = xtest.astype("float32")
ytest = ytest.astype("float32")

xtrain = torch.tensor(xtrain, requires_grad=True)
ytrain = torch.tensor(ytrain.to_numpy(), requires_grad=True).reshape([-1, 1])
xtest = torch.tensor(xtest, requires_grad=True)
ytest = torch.tensor(ytest.to_numpy(), requires_grad=True).reshape([-1, 1])


# Create a DataLoader for batching and shuffling the data
train_dataset = TensorDataset(xtrain, ytrain)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize your neural network
model = Net()

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)

        # Compute loss
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print progress
    if (epoch+1) % print_interval == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

