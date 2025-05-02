import numpy as np
from utils.utils import *
from models.knn.knn import KNN
from models.random_forest.random_forest import RandomForest
from utils.processing import train_val_from_folds

def knn_cross_validation(X_train_folds, y_train_folds, k_values):
    k_accuracies = {}
    k_f1s = {}
    for k in k_values:
        accuracies = []
        f1s = []
        for f, fold in enumerate(X_train_folds):
            # TODO: generic model interface?
            knn = KNN()
            
            # Holdout validation set
            x_train, x_val, y_train, y_val = train_val_from_folds(X_train_folds, y_train_folds, f)
            
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


def random_forest_cross_validation(n_trees,  random_forests: dict[str,RandomForest], X_train_folds, y_train_folds):
    ntree_accuracies = {}
    ntree_f1s = {}
    for ntree in n_trees:
        accuracies = []
        f1s = []
        for f, fold in enumerate(X_train_folds):
            forest = random_forests[ntree]

            # Concatenate all folds except current validation set
            x_train, x_val, y_train, y_val = train_val_from_folds(X_train_folds, y_train_folds, f)
            
            # Train and predict 
            forest.train(x_train, y_train)
            # forest.train(X_train_folds[fold], y_train_folds[fold])
            y_val_predict = forest.predict(x_val)
            
            # Calculate metrics
            tp, fp, tn, fn = confusion_matrix(y_val_predict, y_val)
            val_acc = calc_accuracy(tp, tn, y_val_predict.shape[0])
            precision = calc_precision(tp, fp)
            recall = calc_recall(tp, fn)
            f1 = calc_f1_score(precision, recall)
            print("Ntree", ntree, "fold", fold, "Accuracy", f"{val_acc:.2f}", "F1", f"{f1:.2f}")
            
            # Store results for this fold
            accuracies.append(val_acc)
            f1s.append(f1)
            
        # Store all results for this ntree value
        ntree_accuracies[ntree] = accuracies
        ntree_f1s[ntree] = f1s
            

    return ntree_accuracies, ntree_f1s
