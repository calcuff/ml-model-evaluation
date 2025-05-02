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
from models.decision_tree.split_criteria import InformationGain
from models.decision_tree.stop_criteria import MinimalGain
from models.random_forest.random_forest import RandomForest

# Load digits dataset
digits_dataset_X, digits_dataset_y = load_digits_dataset()
# Normalize data
digits_dataset_X = normalize(digits_dataset_X)

# Split into test and train
X_train, X_test, y_train, y_test = shuffle_and_split(digits_dataset_X, digits_dataset_y)

NUM_FOLDS = 10
X_train_folds, y_train_folds = stratified_folds(X_train, y_train, NUM_FOLDS)

# Number of decision trees in random forest to cross validate
N_TREES = [1,5,10,20,30,40,50]
random_forests = {}
split_metric=InformationGain()
stop_criteria=MinimalGain(minimal_gain=0.1, split_metric=split_metric, probability_threshold=1.0)
for ntree in N_TREES:
    forest = RandomForest(ntree, stop_criteria=stop_criteria, node_split_metric=split_metric)
    random_forests[ntree] = forest