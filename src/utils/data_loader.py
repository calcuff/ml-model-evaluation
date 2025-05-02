from sklearn import datasets

def load_digits_dataset():
    # Load digits dataset
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    return digits_dataset_X, digits_dataset_y