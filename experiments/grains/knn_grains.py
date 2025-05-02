import sys
import os

# Dynamically add the src/ folder to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from utils.validation import knn_cross_validation
from utils.plotting import plot_k_val_results, results_to_csv
from utils.processing import normalize, shuffle_and_split, stratified_folds
from utils.data_loader import load_rice_grains_dataset

X, y = load_rice_grains_dataset()
print("X", X.shape)
print("Y", y.shape)

X = normalize(X)

# Split into test and train
X_train, X_test, y_train, y_test = shuffle_and_split(X, y)

NUM_FOLDS = 10
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, NUM_FOLDS)

k_values = [1, 3, 5, 7, 9, 11, 13, 21, 35, 51, 65]
k_accuracies, k_f1s = knn_cross_validation(X_train_folds, y_train_folds, k_values)

results_dir = "results/"
results_to_csv(k_values, k_accuracies, k_f1s, results_dir + "grains-knn-results.csv")
plot_k_val_results(k_values, k_accuracies, "Accuracy", "Validation Accuracy across K values", results_dir + "grains-knn-val-accuracies.png", "rice grains dataset")
plot_k_val_results(k_values, k_f1s, "F1 Score", "Validation F1 scores across K values", results_dir + "grains-knn-val-f1.png", "rice grains dataset")    
