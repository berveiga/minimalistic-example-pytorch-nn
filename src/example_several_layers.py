""" This code provides a simple example of unidimensional extrapolation using 
a simple feedforward neural network in PyTorch.
Author: Bernardo Veiga (ber.veiga@gmail.com)
"""

# Next step: move the creation of dataframe to be plotted outside the training code
# Next step: move the creation of training data to specific .py script
# Next step (2): move the plotting .py script
# Next step (3): change color of one of the graphs in the test data
# Load libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from nnclasses import MyNN, MyTrainData, MyTestData

# The next command line may be deleted
from plotnine import data, aes, ggplot, geom_point, ggtitle
from plotnine.animation import PlotnineAnimation

# Set random seed
np.random.seed(123)

# To-do add pytorch seed (if required)

"""
 Generate training data
 First example of training data: data to be interpolated
"""
x_vec_train_1 = np.arange(-2, -1, 0.02)
x_vec_train_2 = np.arange(1, 2, 0.02)
x_vec_train = np.concatenate([x_vec_train_1, x_vec_train_2])

# Second example of training data: data to be extrapolated
# x_vec_train = np.arange(-2, 3, 0.05)

x_vec_train = np.reshape(x_vec_train, (-1, 1)).astype("float32")

# Generate target values for the training data
def quad(x):
    return 0.5 * x**2 - x


y_vec_train = np.array(list(map(quad, x_vec_train)))

x_train = torch.tensor(x_vec_train, requires_grad=True)
y_train = torch.tensor(y_vec_train, requires_grad=True)

# Generate test data
x_vec_test = np.arange(-1, 1, 0.02).astype("float32").reshape([-1, 1])
# x_vec_test = np.arange(3, 5, 0.02).astype("float32").reshape([100, 1])
y_vec_test = np.array(list(map(quad, x_vec_test)))

x_test = torch.tensor(x_vec_test, requires_grad=True)
y_test = torch.tensor(y_vec_test, requires_grad=True)

"""
Train the neural network.
"""

my_train_dataset = MyTrainData(x_train, y_train)
my_test_dataset = MyTestData(x_test, y_test)
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
    out_model = model
    return out_model

def generate_train_test_data(inp_epochs):
    """
    Generate dataframes with modelled and true values for training and test data 
    """
    model = fit_neural_network(inp_epochs)

    output_tensor_train = model(my_train_dataset.x)
    output_tensor_test = model(my_test_dataset.x)

    df_model_train =pd.DataFrame({"x":my_train_dataset.x.detach().numpy().reshape([-1, ]),
                            "y":output_tensor_train.detach().numpy().reshape([-1, ]), 
                            "flag":len(output_tensor_train)*["train"]})

    df_model_test =pd.DataFrame({"x":my_test_dataset.x.detach().numpy().reshape([-1, ]),
                            "y":output_tensor_test.detach().numpy().reshape([-1, ]), 
                            "flag":len(output_tensor_test)*["test"]})

    df_true_train=pd.DataFrame({"x":x_vec_train.reshape([-1,]), 
    "y":y_vec_train.reshape([-1,]), 
    "flag":len(x_vec_train)*["true train values"]})

    df_true_test=pd.DataFrame({"x":x_vec_test.reshape([-1,]), 
    "y":y_vec_test.reshape([-1,]), 
    "flag":len(x_vec_test)*["true test values"]})

    df_consolidate_final = pd.concat(
        (df_model_train,
         df_model_test,
         df_true_test,
         df_true_train,
         ), axis=0
    )
    return df_consolidate_final

# Plot modelled test data
EPOCHS = 1000 
df_consolidate = generate_train_test_data(EPOCHS)
fig_1000 = ggplot(df_consolidate, aes(x="x", y="y", color="flag")) + geom_point() + ggtitle(str(EPOCHS) + " epochs")
fig_1000
fig_1000.save("../plots/plot_1000.png")