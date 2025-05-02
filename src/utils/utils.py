import sklearn.utils, sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


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


def confusion_matrix(y_predict, y):
    positive_mask = y == 1
    y_positive = y[positive_mask]
    y_predict_positive = y_predict[positive_mask]
    
    y_negative = y[~positive_mask]
    y_predict_negative = y_predict[~positive_mask]
    
    true_positive = np.sum(y_predict_positive == y_positive)
    false_negative = np.size(y_predict_positive) - true_positive
    
    true_negative = np.sum(y_predict_negative == y_negative)
    false_positive =  np.size(y_predict_negative) - true_negative
    
    # print("------")
    # print("true_positive", true_positive, "false_negative", false_negative, "shape", y_predict_positive.shape[0])
    # print("true_negative", true_negative, "false_positive", false_positive, "shape", y_predict_negative.shape[0])
    return true_positive, false_positive, true_negative, false_negative

def calc_accuracy(tp, tn, total_test_count):
    return (tp+tn)/total_test_count

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def calc_f1_score(precision, recall, beta=1):
    return (1 + beta**2)*(precision * recall)/(beta**2 * precision + recall)

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