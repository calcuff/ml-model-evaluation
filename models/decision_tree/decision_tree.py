import pandas as pd
from decision_tree.node import Node, partition, partition_numerical
from decision_tree.split_criteria import SplitMetric, InformationGain
from decision_tree.stop_criteria import StopCriteria, MinimalGain
from typing import List
import operator
import numpy as np
import math

class DecisionTree():
    def __init__(self, probability_threshold=1.0, split_metric=InformationGain(), attribute_selection="random", stop_criteria=None):
        # Set probability threshold
        self.probability_threshold = probability_threshold
        # Validate correct type for split metric
        if not isinstance(split_metric, SplitMetric):
            raise TypeError(f"Expected an instance of SplitMetric, got {type(split_metric).__name__}")
        self.split_metric:SplitMetric = split_metric
        self.attribute_selection =  attribute_selection
        self.stop_criteria = stop_criteria


    def train(self, X:np.ndarray, y:np.ndarray):
        # Set root node
        self.root_node = Node(X, y, "", "",  list(range(X.shape[1])), None, 0)
        # Recursively build the decision tree
        self.build(self.root_node)


    def build(self, node:Node):
        # Check if we are done
        if self.done(node):
            node.set_as_leaf()
            return
        
        # Find the best attribute to test
        best_attribute_idx, best_mean, max_info_gain = self.find_best_attribute_to_test(node)
        # Mark the outgoing attribute to split on
        node.set_out_attr(best_attribute_idx, best_mean)
        
        # Get child attributes to test based on selection policy
        attributes = self.get_child_attributes(node, best_attribute_idx)
        X, y = node.get_data()
            
        # Build nodes for each value of the best attribute
        if field_is_numeric(X[:, best_attribute_idx]):
            X_less, y_less = partition_numerical(X, y, best_attribute_idx, best_mean, operator.le)
            self.build_child(X_less, y_less, best_attribute_idx, best_mean, attributes, "less", node)
            X_greater, y_greater = partition_numerical(X, y, best_attribute_idx, best_mean, operator.gt)
            self.build_child(X_greater, y_greater, best_attribute_idx, best_mean, attributes, "greater", node)
        else:
            self.build_categorical_split(node, X, y, best_attribute_idx, attributes)
    
                
    def build_categorical_split(self, node, X, y, best_attr_idk, attributes):
        groups = np.unique(X[:, best_attr_idk])
        for g in groups:
            # Partition on attribute and specific value
            X_partition, y_partition = partition(X, y, best_attr_idk, g)
            self.build_child(X_partition, y_partition, in_attr=best_attr_idk, decision_value=g, attributes=attributes, branch_label=g, parent=node)
                

    def build_child(self,X, y, in_attr, decision_value, attributes, branch_label, parent:Node):
        child_node = Node(X, y, in_attr=in_attr, decision_value=decision_value, remaining_attr=attributes, parent=parent, depth=parent.get_depth()+1)
        self.build(child_node)
        parent.add_node(branch_label, child_node)


    def get_child_attributes(self, node:Node, best_attribute:str):
        if self.attribute_selection == "remaining":
            attributes = node.get_remaining_attribtues().copy()
            attributes.remove(best_attribute)
            return attributes
        elif self.attribute_selection == "random":
            X, _ = node.get_data()
            return get_random_attributes(X)
        return None


    def find_best_attribute_to_test(self, node:Node):
        return self.split_metric.compute(node)
    
    
    def done(self, node:Node):
        return self.stop_criteria.done(node=node)


    def classify(self, X: np.ndarray) -> List[str]:
        return self.root_node.classify(X)
    

def get_random_attributes(X: np.ndarray):
    m = int(math.sqrt(X.shape[1]))
    random_columns = np.random.choice(X.shape[1], size=m, replace=False)
    return random_columns.tolist()


def field_is_numeric(col: np.ndarray) -> bool:
    sample = col[0]
    return isinstance(sample, (int, float, np.integer, np.floating))