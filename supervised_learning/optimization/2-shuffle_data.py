#!/usr/bin/env python3
"""Perform optimization"""


import numpy as np


def shuffle_data(X, Y):
    """
    function to shuffle data
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
