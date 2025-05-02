from sklearn import datasets
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.processing import normalize, one_hot_encode

def load_digits_dataset():
    # Load digits dataset
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    return digits_dataset_X, digits_dataset_y

def load_rice_grains_dataset():
    data = np.genfromtxt('../../data/rice.csv', delimiter=',', skip_header=1, dtype=object)
    print(data.shape)
    X = data[:, :-1].astype(float)        # All numeric columns
    y_str = data[:, -1].astype(str)       # Last column as strings

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    return X, y

def load_credit_loan_dataset():
    data = np.genfromtxt('../../data/credit_approval.csv', delimiter=',', skip_header=1, dtype=object, encoding=None)

    categorical_cols = [0, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    numerical_cols = [1, 2, 7, 13, 14]
    
    # Extract numerical features
    X_numerical = data[:, numerical_cols].astype(float)
    X_numerical_norm = normalize(X_numerical)
    print(X_numerical_norm[0])

    # Process categorical columns 
    X_categorical_encoded = one_hot_encode(data, categorical_cols)
    
    print("Encoded shape", X_categorical_encoded.shape)
    
    # Stack cols together
    X = np.hstack((X_numerical_norm, X_categorical_encoded))

    # Labels
    y = data[:, -1].astype(int)
    
    return X, y