import sys
import os

# Dynamically add the src/ folder to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from utils.validation import cross_validate_knn
from utils.plotting import plot_k_val_results
from utils.data_loader import load_digits_dataset
from utils.processing import normalize, shuffle_and_split, stratified_folds

# Load digits dataset
digits_dataset_X, digits_dataset_y = load_digits_dataset()
# Normalize data
digits_dataset_X = normalize(digits_dataset_X)

# Split into test and train
X_train, X_test, y_train, y_test = shuffle_and_split(digits_dataset_X, digits_dataset_y)

NUM_FOLDS = 10
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, NUM_FOLDS)

k_values = [1, 3, 5, 7, 9, 11, 13, 21, 35, 51]
k_accuracies, k_f1s = cross_validate_knn(X_train_folds, y_train_folds, k_values)

plot_k_val_results(k_values, k_accuracies, "Accuracy", "Validation Accuracy across K values", "results/digits-k-val-accuracies.png", "digits dataset")
plot_k_val_results(k_values, k_f1s, "F1 Score", "Validation F1 scores across K values", "results/digits-k-val-f1.png", "digits dataset")    
