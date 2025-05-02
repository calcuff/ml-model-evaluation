import numpy as np
from utils.utils import *
from models.knn.knn import KNN


def cross_validate_knn(X_train_folds, y_train_folds, k_values):
    k_accuracies = {}
    k_f1s = {}
    for k in k_values:
        accuracies = []
        f1s = []
        for f, fold in enumerate(X_train_folds):
            # TODO: generic model interface?
            knn = KNN()
            
            # Holdout validation set
            x_train = np.concatenate([X_train_folds[j] for j in range(len(X_train_folds)) if j != f])
            y_train = np.concatenate([y_train_folds[j] for j in range(len(y_train_folds)) if j != f])
            x_val = X_train_folds[f]
            y_val = y_train_folds[f]
            
            # Train and predict
            knn.train(x_train, y_train)
            y_val_predict = knn.predict(x_val, k=k)
            # Calculate metrics
            tp, fp, tn, fn = confusion_matrix(y_val_predict, y_val)
            val_acc = calc_accuracy(tp, tn, y_val_predict.shape[0])
            precision = calc_precision(tp, fp)
            recall = calc_recall(tp, fn)
            f1 = calc_f1_score(precision, recall)
            # Store results for this fold
            accuracies.append(val_acc)
            f1s.append(f1)
            
        # Store all results for this k value
        k_accuracies[k] = accuracies
        k_f1s[k] = f1s
        
    return k_accuracies, k_f1s