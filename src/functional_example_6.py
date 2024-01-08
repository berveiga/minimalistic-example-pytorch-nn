""" This code provides a simple example of unidimensional extrapolation using 
a simple feedforward neural network in PyTorch.
Author: Bernardo Paschoarelli
"""

# Load libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# The next command line may be deleted
from plotnine import data, aes, ggplot, geom_point
from plotnine.animation import PlotnineAnimation

# Set random seed
np.random.seed(123)

# Generate training data
# First example of training data: data to be interpolated
x_vec_train_1 = np.arange(-2, -1, 0.02)
x_vec_train_2 = np.arange(1, 2, 0.02)
x_vec_train = np.concatenate([x_vec_train_1, x_vec_train_2])

# Second example of training data: data to be extrapolated
#x_vec_train = np.arange(-2, 3, 0.05).astype("float32")

len_train = len(x_vec_train)
x_vec_train = np.reshape(x_vec_train, (len_train, 1))
x_vec_train = x_vec_train.astype("float32")

# Generate target values for the training data
def quad(x):
    return 0.5 * x**2 - x


y_vec_train = np.array(list(map(quad, x_vec_train)))

x_train = torch.tensor(x_vec_train, requires_grad=True)
y_train = torch.tensor(y_vec_train, requires_grad=True)

# Generate test data
x_vec_test = np.arange(-1, 1, 0.02).astype("float32").reshape([100, 1])
# x_vec_test = np.arange(3, 5, 0.02).astype("float32").reshape([100, 1])
y_vec_test = np.array(list(map(quad, x_vec_test)))

x_test = torch.tensor(x_vec_test, requires_grad=True)
y_test = torch.tensor(y_vec_test, requires_grad=True)

# Define training and test dataframe
df_train = pd.DataFrame(
    {
        "x": x_vec_train.reshape(
            [
                -1,
            ]
        ),
        "y": y_vec_train.reshape(
            [
                -1,
            ]
        ),
    }
)

df_test = pd.DataFrame(
    {
        "x": x_vec_test.reshape(
            [
                -1,
            ]
        ),
        "y": y_vec_test.reshape(
            [
                -1,
            ]
        ),
    }
)

# Plot training and test datasets
concatenated_x_data = np.concatenate(
    (
        x_vec_train.reshape(
            [
                -1,
            ]
        ),
        x_vec_test.reshape(
            [
                -1,
            ]
        ),
    )
)

concatenated_y_data = np.concatenate(
    (
        y_vec_train.reshape(
            [
                -1,
            ]
        ),
        y_vec_test.reshape(
            [
                -1,
            ]
        ),
    )
)

flag_y = int(len(concatenated_x_data) / 2) * ["training_set"] + int(
    len(concatenated_x_data) / 2
) * ["test_set"]

df_consolidated_data = pd.DataFrame(
    {"x": concatenated_x_data, "y": concatenated_y_data, "flag": flag_y}
)

ggplot(df_consolidated_data, aes(x="x", y="y", color="flag")) + geom_point()


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


class MyTestData(Dataset):
    def __init__(self, len):
        super().__init__()
        self.x = x_test
        self.y = y_test
        self.len = len

    def __getitem__(self, i):
        z = self.x[i], self.y[i]
        return z

    def __len__(self):
        return self.len


my_train_dataset = MyTrainData(len_train)
my_test_dataset = MyTestData(100)
dataloader = DataLoader(my_train_dataset, batch_size=1)


def fit_neural_network(EPOCHS):
    model = MyNN()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    list_loss = []

    for epoch in range(EPOCHS):
        sum_loss = 0
        for u, v in dataloader:
            output = model(u)
            loss_increment = loss_fn(v, output)
            optimizer.zero_grad()
            loss_increment.backward()
            optimizer.step()
            sum_loss += loss_increment
            list_loss.append(float(sum_loss.detach().numpy()))
        if epoch % 50 == 0:
            print("------------")
            print("Step number " + str(epoch))
            print(sum_loss)

    output_tensor_train = model(my_train_dataset.x)
    output_vec_train = (
        output_tensor_train.detach()
        .numpy()
        .reshape(
            [
                -1,
            ]
        )
    )

    output_tensor_test = model(my_test_dataset.x)
    output_vec_test = (
        output_tensor_test.detach()
        .numpy()
        .reshape(
            [
                -1,
            ]
        )
    )

    # Concatenate training and test data
    print(str(len(output_vec_test)))
    concatenated_x_train_test = np.concatenate(
        (
            x_vec_train.reshape(
                [
                    -1,
                ]
            ),
            x_vec_test.reshape(
                [
                    -1,
                ]
            ),
        )
    )
    concatenated_y_train_test = np.concatenate(
        (
            y_vec_train.reshape(
                [
                    -1,
                ]
            ),
            output_vec_test,
        )
    )
    flag_y_train_test = int(len(concatenated_x_train_test) / 2) * ["train_data"] + int(
        len(concatenated_y_train_test) / 2
    ) * ["test_data"]
    df_consolidated_train_test = pd.DataFrame(
        {
            "x": concatenated_x_train_test,
            "y": concatenated_y_train_test,
            "flag": flag_y_train_test,
        }
    )

    concatenated_x_train_test2 = np.concatenate(
        (
            x_vec_train.reshape(
                [
                    -1,
                ]
            ),
            x_vec_test.reshape(
                [
                    -1,
                ]
            ),
        )
    )
    concatenated_y_train_test2 = np.concatenate(
        (
            output_vec_train,
            y_vec_test.reshape(
                [
                    -1,
                ]
            ),
        )
    )
    flag_y_train_test2 = int(len(concatenated_x_train_test2) / 2) * [
        "output_train_data"
    ] + int(len(concatenated_y_train_test2) / 2) * ["test_data"]

    df_consolidated_train_test2 = pd.DataFrame(
        {
            "x": concatenated_x_train_test2,
            "y": concatenated_y_train_test2,
            "flag": flag_y_train_test2,
        }
    )
    df_consolidated_final = pd.concat(
        (df_consolidated_train_test, df_consolidated_train_test2), axis=0
    )
    return df_consolidated_final
    # Plot modelled test data


df_consolidated_50 = fit_neural_network(50)
fig_50 = ggplot(df_consolidated, aes(x="x", y="y", color="flag")) + geom_point()
fig
fig.save("../plots/plot_50.png")

df_consolidated_5000 = fit_neural_network(5000)
fig_5000 = ggplot(df_consolidated_5000, aes(x="x", y="y", color="flag")) + geom_point()
fig_5000
fig_5000.save("../plots/plot_5000.png")

def plot_neural_network(k):
    p = (ggplot(df_consolidated, aes(x="x", y="y", color="flag")) + geom_point())
    return p