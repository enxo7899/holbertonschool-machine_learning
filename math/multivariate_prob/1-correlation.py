#!/usr/bin/env python3
"""Correlation Task 1"""


import numpy as np


def correlation(C):
    """
    Function for correlation
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    d = C.shape[0]
    diag = np.diag(C)
    diag_sqrt = np.sqrt(diag)
    outer_sqrt = np.outer(diag_sqrt, diag_sqrt)
    correlation_matrix = C / outer_sqrt
    return correlation_matrix
