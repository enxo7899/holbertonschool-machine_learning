#!/usr/bin/env python3
"""Correlation Task 1"""


import numpy as np


def correlation(C):
    """
    Function for correlation
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    D = np.diag(1 / np.sqrt(np.diag(C)))
    return np.dot(np.dot(D, C), D)
