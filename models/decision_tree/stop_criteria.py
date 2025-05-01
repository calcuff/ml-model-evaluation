from abc import ABC, abstractmethod
from decision_tree.node import Node
from decision_tree.split_criteria import SplitMetric

class StopCriteria(ABC):
    @abstractmethod
    def done(self, node: Node) -> bool:
        pass
    
    
class MinimalSizeForSplit(StopCriteria):
    """
    Minimal Size For Split Stop Criteria
    
    Will stop when the number of instances in
    a node is less than or equal to the 
    specified minimal amount, or when the 
    probability distribution is greater than 
    or equal to the specified threshold.
    """
    def __init__(self, n:int, probability_threshold=1.0):
        self.n = n
        self.p = probability_threshold
        
    def done(self, node:Node):
        X, _ = node.get_data()
        _, p = node.get_majority_probability()
        return p >= self.p or X.shape[0] <= self.n
    

class MinimalGain(StopCriteria):
    """
    Minimal Gain Stop Criteria
    
    Will stop when information gain is less
    than or equal to the specified minimal 
    information gain, or when the probability
    distribution is greater than or equal to
    the specified threshold.
    """
    def __init__(self, minimal_gain: float, split_metric:SplitMetric, probability_threshold=1.0):
        self.minimal_gain = minimal_gain
        self.split_metric = split_metric
        self.p = probability_threshold
    
    def done(self, node: Node):
        # TODO: we will do this twice
        # If we were to split on this node, do we get enough information gain
        _, p = node.get_majority_probability()
        _, _, max_info_gain = self.split_metric.compute(node)
        return  p >= self.p or max_info_gain <= self.minimal_gain
        
        
        
class MaximalDepth(StopCriteria):
    """
    Maximal Depth Stop Criteria
    
    Will stop when a node's depth exceeds the
    specified maximal depth, or when the probability
    distribution is greater than or equal to
    the specified threshold.
    """
    
    def __init__(self, max_depth:int, probability_threshold=1.0):
        self.max_depth = max_depth
        self.p = probability_threshold
        
    def done(self, node: Node):
        _, p = node.get_majority_probability()
        return p >= self.p or node.get_depth() >= self.max_depth