#!/usr/bin/env python3
"""Perform binary classification with one hidden layer"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Arguments:
    Y -- a numpy.ndarray with shape (m,) containing numeric class labels
    classes -- the maximum number of classes found in Y

    Returns:
    a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y) < 1 \
            or not isinstance(classes, int) or classes < 1:
        return None

    m = Y.shape[0]
    Y_one_hot = np.zeros((classes, m))

    for i in range(m):
        c = Y[i]
        if c < 0 or c >= classes:
            return None
        Y_one_hot[c, i] = 1

    return Y_one_hot
