import sys
import os

# Dynamically add the src/ folder to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

import numpy as np
from utils.processing import normalize, shuffle_and_split, stratified_folds
from utils.metrics import *
from utils.validation import test_decision_tree
from utils.plotting import plot_dt_accuracy_histogram
from models.decision_tree.decision_tree import DecisionTree
from models.decision_tree.stop_criteria import MinimalGain


data = np.genfromtxt('../../data/parkinsons.csv', delimiter=',', skip_header=1, dtype=float)
print(data.shape)
X = data[:, :-1]
y = data[:, -1]

print("X", X.shape)
print("Y", y.shape)

X = normalize(X)

X_train, X_test, y_train, y_test = shuffle_and_split(X, y)
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, 5)


# Best Decision Tree hyper paremeters
best_stop_criteria = MinimalGain(0.01)
best_decision_tree = DecisionTree(stop_criteria=best_stop_criteria)
train_accuracies, test_accuracies = test_decision_tree(X_train, X_test, y_train, y_test, best_decision_tree, 100)

results_dir = "results/"
plot_dt_accuracy_histogram(train_accuracies, type="Training", image_name=results_dir + "best_dt_parkinsons_training_hist.png", sub_title="Parkinsons dataset")
plot_dt_accuracy_histogram(test_accuracies, type="Testing", image_name=results_dir + "best_dt_parkinsons_testing_hist.png", sub_title="Parkinsons dataset")


