from sklearn import datasets
import numpy as np
from sklearn.preprocessing import LabelEncoder

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