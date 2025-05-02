import sys
import os

# Dynamically add the src/ folder to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

import numpy as np
from utils.processing import normalize, shuffle_and_split, stratified_folds
from utils.metrics import *
from models.neural_net.neural_network import NeuralNetwork
from utils.plotting import plot_nn_loss_curve

data = np.genfromtxt('../../data/parkinsons.csv', delimiter=',', skip_header=1, dtype=float)
print(data.shape)
X = data[:, :-1]
y = data[:, -1]
y = y.flatten().astype(int)
print("X", X.shape)
print("Y", y.shape)

X = normalize(X)

X_train, X_test, y_train, y_test = shuffle_and_split(X, y)
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, 5)

# Best Neural Network
lr = 5e-1
reg = 0.0
hidden_dims = [30,15]
nn = NeuralNetwork(input_dim=22, hidden_layer_dims=hidden_dims, output_dim=1, reg=reg, lr=lr)

# Tracking variables
train_losses = []
test_losses = []
x_seen = []

num_epochs = 50
for epoch in range(1, num_epochs + 1):
    # train
    nn.train(X_train, y_train, num_iters=1)

    # training loss
    train_loss, _ = nn.loss(X_train, y_train)
    # test loss
    test_loss, _ = nn.loss(X_test, y_test)

    # Record losses
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    x_seen.append(epoch * len(X_train))


plot_nn_loss_curve(x_seen, train_losses, test_losses, "results/best_nn_parkinsons_loss_curve.png")
