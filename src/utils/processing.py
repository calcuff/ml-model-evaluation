import sklearn.utils, sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple
import numpy as np

# min-max normalization
def normalize(X):
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    
    r = x_max - x_min
    # Avoid division by zero
    r[r == 0] = 1
    
    X_norm = (X - x_min)/r
    return X_norm

# Shuffle data and split into training + test
def shuffle_and_split(X, y):
    # Shuffle data set
    X, y = sklearn.utils.shuffle(X,y)
    # Partition data into train + test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, train_size=0.8, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test

# Split data into num stratified folds
def stratified_folds(X:np.ndarray, y:np.ndarray, num_folds:int)-> Tuple[list, list]:
    # Shuffle data set
    X, y = sklearn.utils.shuffle(X,y)
    
    X_folds = []
    y_folds = []
    for i in range(num_folds, 1, -1):
        # Split into train of size 1/i. Proportion will grow each iteration as fold data is removed from original
        X_train, X_rest, y_train, y_rest = sklearn.model_selection.train_test_split(X, y, train_size=1/i, stratify=y)
        # Store data for this fold
        X_folds.append(X_train)
        y_folds.append(y_train)
        
        X, y = X_rest, y_rest

    # We have removed n-1 folds from the original, our last fold is what's left
    X_folds.append(X)
    y_folds.append(y)
    return X_folds, y_folds

def one_hot_encode(data: np.ndarray, columns: list):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    total_cols = 0
    for c in columns:
        unique_vals = np.unique(data[:, c])
        total_cols += unique_vals.shape[0]

    encoded_data = np.zeros((data.shape[0], total_cols))

    start = 0
    for i, c in enumerate(columns):
        d = data[:, c].reshape(-1, 1)
        transformed = ohe.fit_transform(d)
        end = start + transformed.shape[1]
        encoded_data[:, start:end] = transformed
        start = end

    return encoded_data


def train_val_from_folds(X_train_folds, y_train_folds, f):
    x_train = np.concatenate([X_train_folds[j] for j in range(len(X_train_folds)) if j != f])
    y_train = np.concatenate([y_train_folds[j] for j in range(len(y_train_folds)) if j != f])
    x_val = X_train_folds[f]
    y_val = y_train_folds[f]
    return x_train, x_val, y_train, y_val