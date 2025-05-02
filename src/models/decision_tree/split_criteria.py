from models.decision_tree.node import Node, partition, partition_numerical
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import operator
import random

class SplitMetric(ABC):
    @abstractmethod
    def compute(self, node: Node) -> Tuple[int, float, float]:
        pass

class InformationGain(SplitMetric):
    def __init__(self):
        pass
        
    def compute(self, node:Node) -> Tuple[int, float, float]:
        X, y = node.get_data()
        parent_entropy = self.calculate_entropy(y)
        attributes = self.get_attributes_to_test(node)
        
        if node.isEmpty():
            return random_attribute(attributes), 0, 0
        
        max_info_gain = -float('inf')
        best_attr = None
        best_mean = None
        mean = None
        
        # Get the information gain of each attribute
        for attr_idx in attributes:
            sample_val = X[0, attr_idx]
            if isinstance(sample_val, (int, float, np.integer, np.floating)):
                info_gain, mean = self.information_gain_numerical_avg(X, y, attr_idx, parent_entropy)
            else:
                info_gain = self.information_gain_categorical(X, y, attr_idx, parent_entropy)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attr = attr_idx
                best_mean = mean

        return best_attr, best_mean, max_info_gain
    
    # TODO: this takes too long to be useful
    def information_gain_numerical_dynamic(self, X:np.ndarray, y:np.ndarray, attr:str, parent_entropy:float):
        values = X[attr].sort_values().unique()
        max_info_gain = -float('inf')
        best_mean = 0.0
        
        if len(values) == 1:
            return 0.0, values[0]
        
        for i in range(len(values) - 1):
            mean = (values[i] + values[i + 1]) / 2
            I = self.sub_partition_entropy_numerical(X, y, attr, mean)
            information_gain = parent_entropy - I
            if information_gain > max_info_gain:
                max_info_gain = information_gain
                best_mean = mean
                
        return max_info_gain, best_mean
    
    
    def information_gain_numerical_avg(self, X, y, attr_idx:int, parent_entropy:float):
        mean = np.mean(X[:, attr_idx])
        I = self.sub_partition_entropy_numerical(X, y, attr_idx, mean)
        information_gain = parent_entropy - I
        return information_gain, mean
    
    
    def information_gain_categorical(self, X, y, attr_idx:int, parent_entropy:float):
        groups = np.unique(X[:, attr_idx])
        # Entropy of partitions based off attribute
        I = 0.0
        for attr_value in groups:
            I += self.sub_partition_entropy(X, y, attr_idx, attr_value)
        information_gain = parent_entropy - I
        return information_gain
    
    
    def get_attributes_to_test(self, node:Node):
        return node.get_remaining_attribtues()


    def sub_partition_entropy(self, X, y, attr, attr_value):
        # Get the X,y partitions based on value of an attribute
        X_partition, y_partition = partition(X, y, attr, attr_value)
        # Calculate the entropy of the resulting partition
        partition_entropy = self.calculate_entropy(y_partition)
        # Scale the entropy by its proportion
        scaled_entropy = X_partition.shape[0]/y.shape[0] * partition_entropy
        
        return scaled_entropy

    def sub_partition_entropy_numerical(self, X, y, attr, attr_value):
        X_less, y_less = partition_numerical(X, y, attr, attr_value, operator.le)
        partition_entropy_less = self.calculate_entropy(y_less)
        # Scale the entropy by its proportion
        scaled_entropy_less = X_less.shape[0]/y.shape[0] * partition_entropy_less
        
        X_more, y_more = partition_numerical(X, y, attr, attr_value, operator.gt)
        partition_entropy_more = self.calculate_entropy(y_more)
        # Scale the entropy by its proportion
        scaled_entropy_more = X_more.shape[0]/y.shape[0] * partition_entropy_more
        
        return scaled_entropy_less + scaled_entropy_more
    
    def calculate_entropy(self, y:np.ndarray):
        # Get the probability distribution and extract np array
        _, counts = np.unique(y, return_counts=True)
        # Convert counts to probabilities
        probs = counts / counts.sum()
        # Compute entropy, guarding against log(0)
        I = -np.sum(probs * np.log2(probs, where=(probs != 0)))
        return I

def random_attribute(attributes:list):
    random_idx = random.choice(list(range(len(attributes))))
    return attributes[random_idx]

class GiniCriterion(SplitMetric):
    def compute(self, node: Node) -> Tuple[int, float, float]:
        # parent_gini = self.gini(node.get_probabilities())
        X, y = node.get_data()
        min_gini_coefficient = 1.0
        for attr in node.get_remaining_attribtues():
            attribute_gini = 0.0
            groups = np.unique(X[:,attr])
            for attr_value in groups:
                # Get the X,y partitions based on value of an attribute
                X_partition, y_partition = partition(X, y, attr, attr_value)
                # Make a node representing the partition
                # child_node = Node(X_partition, y_partition, attr, attr_value, [])
                # Get the probability distribution and extract np array
                _, counts = np.unique(y_partition, return_counts=True)
                # Convert counts to probabilities
                probs = counts / counts.sum()
                # Child Gini
                child_gini = self.gini(probs)
                # Scale with proportion of partition
                scaled_child_gini =  X_partition.shape[0]/y.shape[0] * child_gini
                # Overall gini for splitting on the attribute
                attribute_gini += scaled_child_gini
            
            # Track lowest Gini + attribute
            if attribute_gini < min_gini_coefficient:
                min_gini_coefficient = attribute_gini
                best_attr = attr

        return best_attr, min_gini_coefficient, None


    def gini(self, probs: list):
        return 1 - np.sum(np.square(probs))