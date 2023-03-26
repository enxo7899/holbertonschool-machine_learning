#!/usr/bin/env python3
"""Meand and Covariance"""


import numpy as np


def mean_cov(X):
    """
    Performsa a finction to calculate mean and covariance
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    centered_X = X - mean
    cov = np.dot(centered_X.T, centered_X) / (n - 1)
    return mean, cov
