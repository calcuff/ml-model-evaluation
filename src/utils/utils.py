import numpy as np

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
    
    return true_positive, false_positive, true_negative, false_negative

def calc_accuracy(tp, tn, total_test_count):
    return (tp+tn)/total_test_count

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def calc_f1_score(precision, recall, beta=1):
    return (1 + beta**2)*(precision * recall)/(beta**2 * precision + recall)
