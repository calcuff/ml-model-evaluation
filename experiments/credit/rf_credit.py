import sys
import os

# Dynamically add the src/ folder to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from utils.validation import random_forest_cross_validation
from utils.plotting import plot_k_val_results, results_to_csv
from utils.data_loader import load_credit_loan_dataset
from utils.processing import normalize, shuffle_and_split, stratified_folds
from models.decision_tree.split_criteria import InformationGain
from models.decision_tree.stop_criteria import MinimalGain
from models.random_forest.random_forest import RandomForest
import numpy as np

# Load credit loan dataset
X, y = load_credit_loan_dataset()


# Split into test and train
X_train, X_test, y_train, y_test = shuffle_and_split(X, y)

NUM_FOLDS = 10
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, NUM_FOLDS)

# Number of decision trees in random forest to cross validate
N_TREES = [1,5,10,20,30,40,50,100]
random_forests = {}
split_metric=InformationGain()
stop_criteria=MinimalGain(minimal_gain=0.001, split_metric=split_metric, probability_threshold=1.0)
for ntree in N_TREES:
    forest = RandomForest(ntree, stop_criteria=stop_criteria, node_split_metric=split_metric, attribute_selection="random")
    random_forests[ntree] = forest

ntree_accuracies, ntree_f1s  = random_forest_cross_validation(N_TREES, random_forests, X_train_folds, y_train_folds)

results_dir = "results/"
results_to_csv(N_TREES, ntree_accuracies, ntree_f1s, results_dir + "credit-rf-ntree-results.csv")
plot_k_val_results(N_TREES, ntree_accuracies, "Accuracy", "Validation Accuracy across Ntree values", results_dir + "credit-rf-ntree-val-accuracies.png", "credit dataset", "Ntree")
plot_k_val_results(N_TREES, ntree_f1s, "F1 Score", "Validation F1 scores across Ntree values", results_dir + "credit-rf-ntree-val-f1.png", "credit dataset", "Ntree")    