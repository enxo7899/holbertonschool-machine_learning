#!/usr/bin/env python3
"""Performs a function that represents a multivariate normal distribution"""

import numpy as np


class MultiNormal:
    """
    MultiNormal Class
    """
    def __init__(self, data):
        """
        A function for multi normal
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot(data-self.mean, (data-self.mean).T)/(data.shape[1]-1)
