import sys
import os

# Dynamically add the src/ folder to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

import numpy as np
from utils.processing import normalize, shuffle_and_split, stratified_folds
from utils.metrics import *
from utils.validation import decision_tree_cross_validation
from utils.plotting import decision_tree_rsults_to_csv, plot_dt_results
from models.decision_tree.decision_tree import DecisionTree
from models.decision_tree.split_criteria import InformationGain, GiniCriterion
from models.decision_tree.stop_criteria import MaximalDepth, MinimalGain, MinimalSizeForSplit
from utils.data_loader import load_rice_grains_dataset

X, y = load_rice_grains_dataset()
print("X", X.shape)
print("Y", y.shape)

X = normalize(X)

X_train, X_test, y_train, y_test = shuffle_and_split(X, y)
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, 5)


# Hyper parameter search on Stop Criteria
stop_criteria = [MaximalDepth(5), MaximalDepth(15), MinimalGain(0.1), MinimalGain(0.01), MinimalSizeForSplit(5), MinimalSizeForSplit(11)]
accuracies, f1s = decision_tree_cross_validation(X_train_folds, y_train_folds, stop_criteria)

results_dir = "results/"
results_csv = results_dir + "grains-dt-results.csv"
results_img = results_dir + "grains-dt-results.png"
decision_tree_rsults_to_csv(accuracies, f1s, results_csv)
plot_dt_results(results_csv,"Rice Grains", results_img)