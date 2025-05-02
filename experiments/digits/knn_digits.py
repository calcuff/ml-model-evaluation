import sys
import os

# Dynamically add the src/ folder to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from utils.validation import knn_cross_validation
from utils.plotting import plot_k_val_results, results_to_csv
from utils.data_loader import load_digits_dataset
from utils.processing import normalize, shuffle_and_split, stratified_folds
import numpy as np

# Load digits dataset
digits_dataset_X, digits_dataset_y = load_digits_dataset()
# Normalize data
digits_dataset_X = normalize(digits_dataset_X)

# Split into test and train
X_train, X_test, y_train, y_test = shuffle_and_split(digits_dataset_X, digits_dataset_y)

NUM_FOLDS = 10
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, NUM_FOLDS)

k_values = [1, 3, 5, 7, 9, 11, 13, 21, 35, 51]
k_accuracies, k_f1s = knn_cross_validation(X_train_folds, y_train_folds, k_values)

results_dir = "results/"
results_to_csv(k_values, k_accuracies, k_f1s, results_dir + "digits-knn-results.csv")
plot_k_val_results(k_values, k_accuracies, "Accuracy", "Validation Accuracy across K values", results_dir + "digits-knn-val-accuracies.png", "digits dataset")
plot_k_val_results(k_values, k_f1s, "F1 Score", "Validation F1 scores across K values", results_dir + "digits-knn-val-f1.png", "digits dataset")    
