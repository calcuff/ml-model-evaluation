from decision_tree.decision_tree import DecisionTree
from decision_tree.split_criteria import SplitMetric
from decision_tree.stop_criteria import StopCriteria
from typing import List
import numpy as np

class RandomForest():
    def __init__(self, ntree:int, stop_criteria:StopCriteria, node_split_metric:SplitMetric):
        self.trees:List[DecisionTree] = []
        for i in range(ntree):
            self.trees.append(DecisionTree(split_metric=node_split_metric, stop_criteria=stop_criteria))

            
    def train(self, X:np.ndarray, y:np.ndarray) -> None:
        # TODO: threading
        for dt in self.trees:
            X_bootstrap, y_bootstrap = self.bootstrap_data(X, y)
            dt.train(X_bootstrap, y_bootstrap)
    
    
    def bootstrap_data(self, X:np.ndarray, y:np.ndarray):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap
        
        
    def predict(self, X: np.ndarray):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, dt in enumerate(self.trees):
            predictions[:,i] = dt.classify(X)
        return self.majority(predictions)
    
    
    def majority(self, predictions:np.ndarray) -> str:
        N, D = predictions.shape
        majority_predict = np.zeros((N))
        # Get unique values and their counts
        for n in range(N):
            values, counts = np.unique(predictions[n], return_counts=True)
            majority_predict[n] = values[np.argmax(counts)]
        return majority_predict