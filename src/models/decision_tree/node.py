from __future__ import annotations
from typing import List
import operator
import numpy as np

class Node():
    def __init__(self, X: np.ndarray, y:np.ndarray, in_attr:str, decision_value:str, remaining_attr:list, parent:Node, depth=0):
        self.classes = np.unique(y)
        self.X = X # data
        self.y = y # labels
        self.in_attr = in_attr # incoming split attribute
        self.in_attr_value = decision_value # incoming attribute value
        self.remaining_attr = remaining_attr # remaining attributes to test
        self.branches: dict[int, Node] = {} # outgoing branches
        self.is_leaf = False  # if this is a leaf node
        self.depth = depth # depth of tree at this node
        self.out_attr =  None # outgoing best attribute
        self.parent = parent # parent node

    def get_data(self):
        return self.X, self.y
    
    def add_node(self, attr_value:str, node):
        self.branches[attr_value] = node

    def classify(self, X: np.ndarray) -> List[str]:
        predictions = []
        for i in range(X.shape[0]):
            prediction = self.classify_instance(X[i])
            predictions.append(prediction)
        return predictions

    def isEmpty(self):
        return self.X.shape[0] == 0

    def predict(self):
        if self.isEmpty():
            return self.parent.predict()
        max_class, _ = self.get_majority_probability()
        return max_class
    
    def classify_instance(self, X: np.ndarray)->str:
        if self.is_leaf:
            return self.predict()
        
        elif is_numerical(X[self.out_attr]):
            return self.follow_comparator_branch(X)
        
        elif self.is_known_attribute_val(X[self.out_attr]):
            return self.branches[X[self.out_attr]].classify_instance(X)
        
        else: # New categorical attribute value we havent see in training
            return self.predict()

    def get_remaining_attribtues(self):
        return self.remaining_attr
    
    def set_out_attr(self, out_attr_idx:int, best_mean:float):
        self.out_attr = out_attr_idx
        self.best_mean = best_mean

    def set_as_leaf(self):
        self.is_leaf = True
        self.set_class_prediction()

    def set_class_prediction(self):
        max_class, max_prob = self.get_majority_probability()
        self.prediction = max_class

    def get_majority_probability(self):
        if self.isEmpty():
            return None, 1.0
        
        values, counts = np.unique(self.y, return_counts=True)
        max_index = np.argmax(counts)
        max_class = values[max_index]
        max_probability = counts[max_index] / counts.sum()
        return max_class, max_probability

    def follow_comparator_branch(self, X):
        if X[self.out_attr] <= self.best_mean:
            return self.branches["less"].classify_instance(X)
        else:
            return self.branches["greater"].classify_instance(X)

    def is_known_attribute_val(self, val):
        return val in self.branches

    def get_probabilities(self):
        probs = []
        values, counts = np.unique(self.y, return_counts=True)
        for c in counts:
            prob = c/counts.sum()
            probs.append(prob)
        return probs
    
    def get_depth(self) -> int:
        return self.depth
    
# Partition X, y on specific value of an attribute
def partition(X:np.ndarray, y:np.ndarray, attr_idx, attr_value):
   # partition mask
    mask = X[:, attr_idx] == attr_value
    # partition X and y
    X_partition = X[mask]
    y_partition = y[mask]
    
    return X_partition, y_partition

def partition_numerical(X:np.ndarray, y:np.ndarray, attr_idx, attr_value, comparison:operator):
    # Apply the comparison function (e.g., operator.le or operator.gt)
    mask = comparison(X[:, attr_idx], attr_value)
    
    # Use the mask to extract the partition
    X_partition = X[mask]
    y_partition = y[mask]
    return X_partition, y_partition

def is_numerical(val):
    return np.issubdtype(type(val), np.number)