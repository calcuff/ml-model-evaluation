import numpy as np
from models.knn.knn import KNN
from models.decision_tree.decision_tree import DecisionTree
from models.random_forest.random_forest import RandomForest
from models.neural_net.neural_network import NeuralNetwork
from utils.processing import train_val_from_folds
from utils.metrics import *

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
            
            # Store results for this fold
            accuracies.append(val_acc)
            f1s.append(f1)
            
        # Store all results for this ntree value
        print("Ntree", ntree, "Accuracy", f"{np.mean(accuracies):.2f}", "F1", f"{np.mean(f1s):.2f}")
        ntree_accuracies[ntree] = accuracies
        ntree_f1s[ntree] = f1s
            

    return ntree_accuracies, ntree_f1s

def decision_tree_cross_validation(X_train_folds, y_train_folds, stop_criteria):
    stop_criteria_accuracies = {}
    stop_criteria_f1s = {}
    for sc in stop_criteria:
        accuracies = []
        f1s = []
        for f, fold in enumerate(X_train_folds):
            decision_tree = DecisionTree(stop_criteria=sc)

            # Concatenate all folds except current validation set
            x_train, x_val, y_train, y_val = train_val_from_folds(X_train_folds, y_train_folds, f)
            
            # Train and predict 
            decision_tree.train(x_train, y_train)
            # forest.train(X_train_folds[fold], y_train_folds[fold])
            y_val_predict = decision_tree.classify(x_val)
            
            # Calculate metrics
            tp, fp, tn, fn = confusion_matrix(y_val_predict, y_val)
            val_acc = calc_accuracy(tp, tn, y_val_predict.shape[0])
            precision = calc_precision(tp, fp)
            recall = calc_recall(tp, fn)
            f1 = calc_f1_score(precision, recall)
            
            # Store results for this fold
            accuracies.append(val_acc)
            f1s.append(f1)
            
        # Store all results for this ntree value
        print("StopCriteria", str(sc), "Accuracy", f"{np.mean(accuracies):.2f}", "F1", f"{np.mean(f1s):.2f}")
        stop_criteria_accuracies[str(sc)] = accuracies
        stop_criteria_f1s[str(sc)] = f1s
            

    return stop_criteria_accuracies, stop_criteria_f1s
    
def neural_network_cross_validation(X_train_folds, y_train_folds, lrs, regs, input_dim, hidden_dims, output_dims):
    results = []

    for lr in lrs:
        for reg in regs:
            for hd in hidden_dims:
                acc_scores = []
                f1_scores = []
                
                for f, fold in enumerate(X_train_folds):
                    nn = NeuralNetwork(input_dim=input_dim, hidden_layer_dims=hd, output_dim=output_dims, reg=reg, lr=lr)
                    # Concatenate all folds except current validation set
                    x_train, x_val, y_train, y_val = train_val_from_folds(X_train_folds, y_train_folds, f)
                    nn.train(x_train, y_train, 2000, verbose=False)
                    y_pred = nn.predict(x_val)
                    tp, fp, tn, fn = confusion_matrix(y_pred, y_val)
                    acc = calc_accuracy(tp, tn, y_val.shape[0])
                    f1 = calc_f1_score(calc_precision(tp,fp), calc_recall(tp,fn))
                    acc_scores.append(acc)
                    f1_scores.append(f1)
                    
                #Store result summary for this hyperparameter combo
                print("LR", lr, "Reg", reg, "HD", str(hd), "Acc", np.mean(acc_scores), "F1",np.mean(f1_scores) )
                results.append({
                    "learning_rate": lr,
                    "regularization": reg,
                    "hidden_dims": str(hd),
                    "mean_accuracy": np.mean(acc_scores),
                    "mean_f1": np.mean(f1_scores)
                })
                
    return results
                

def test_decision_tree(X_train, X_test, y_train, y_test, decision_tree:DecisionTree, iter):
    train_accuracies = []
    test_accuracies = []
    for _ in range(iter):
        # Build decision tree on training data
        dt = decision_tree
        dt.train(X_train, y_train)

        # Classify training set
        y_train_predict = dt.classify(X_train)
        tp, fp, tn, fn = confusion_matrix(y_train_predict, y_train)
        train_acc = calc_accuracy(tp, tn, y_train_predict.shape[0])
        precision = calc_precision(tp, fp)
        recall = calc_recall(tp, fn)
        f1 = calc_f1_score(precision, recall)
        train_accuracies.append(train_acc)
        
        # Classify test set
        y_test_predict = dt.classify(X_test)
        tp, fp, tn, fn = confusion_matrix(y_test_predict, y_test)
        test_acc = calc_accuracy(tp, tn, y_test_predict.shape[0])
        precision = calc_precision(tp, fp)
        recall = calc_recall(tp, fn)
        f1 = calc_f1_score(precision, recall)
        test_accuracies.append(test_acc)

    return train_accuracies, test_accuracies