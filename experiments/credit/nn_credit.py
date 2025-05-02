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
from utils.validation import neural_network_cross_validation
import pandas as pd
from utils.plotting import plot_nn_results
from utils.data_loader import load_credit_loan_dataset

# Load credit loan dataset
X, y = load_credit_loan_dataset()
print("X", X.shape)

X_train, X_test, y_train, y_test = shuffle_and_split(X, y)
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, 5)

lrs = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
regs = [0.0, 1e-3, 1e-2, 1e-1]
hidden_dims = [[2], [2,2], [2,2,2], [10], [10,10], [20], [40], [20,10], [30,15], [40,20,10]]

results  = neural_network_cross_validation(X_train_folds, y_train_folds, lrs, regs, input_dim=68, hidden_dims=hidden_dims, output_dims=1)
results_df = pd.DataFrame(results)

results_csv = "results/credit-nn-results.csv"
results_df.to_csv(results_csv, index=False)

plot_nn_results(results_csv, "Credit", "results/credit-nn-results.png")
