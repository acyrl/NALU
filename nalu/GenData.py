"""
Various helper functions to generate training sets for the NALU and NAC.
"""

import numpy as np


class GenerateDatasetHelper():
    """
    
    Simple helper class to the functions that generate NALU training sets. 
    
    """

    @staticmethod 
    def generate_result(X, op=None):
        if not op:
            return np.sum(X, axis=1, keepdims=True)
        else:
            return op(X)

    @staticmethod
    def sum_func(X):
        return 


def gd_uniform(lower_bound=0, upper_bound=5, size=1000, calc_output=True, op=None, depth=2):
    """
    
    Generates data (a, b, f(a, b)), where a and b are draw independetly from
    the uniform distribution on [lower_bound, upper_bound]. If op is not defined, 
    f(a,b) = a+b.
    
    """ 
    X = np.random.uniform(lower_bound, upper_bound, (size, depth))        
    Y = GenerateDatasetHelper.generate_result(X, op)

    return X, Y

def gd_paper(lower_bound=0, upper_bound=5, size=1000, op=None):
    """
    
    Generates data as described in Appendix B of arXiv:1808.00508v1
    We generate a vector of 100 elements, then choose two contiguous subsequences
    if this array, say a and b, and then return (a,b,f(a,b)).
    
    """
    x_size = 100
    x = np.random.uniform(lower_bound, upper_bound, x_size)

    # There is a better way of doing this.
    sub_seq_indexes = np.random.randint(1, x_size+1, size=(size,2,2))
    sub_seq_indexes_min = np.min(sub_seq_indexes, axis=2, keepdims=True)
    sub_seq_indexes_max = np.max(sub_seq_indexes, axis=2, keepdims=True)
    X_list = []
    

    for i in range(size):
        a = np.sum(x[sub_seq_indexes_min[i][0][0]:sub_seq_indexes_max[i][0][0]])
        b = np.sum(x[sub_seq_indexes_min[i][1][0]:sub_seq_indexes_max[i][1][0]])
        X_list.append((a,b))
        
    X = np.array(X_list)
    Y = GenerateDatasetHelper.generate_result(X, op)
    
    return X, Y