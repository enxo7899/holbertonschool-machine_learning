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
    def pdf(self, x):
        """
        A function for pdf
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.ndim != 2 or x.shape[1] != 1 or x.shape[0] != self.mean.shape[0]:
            raise ValueError(f"x must have the shape ({self.mean.shape[0]}, 1)")
        d = self.mean.shape[0]
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        norm_const = 1.0/((2*np.pi)**(d/2)*np.sqrt(det))
        x_mean = x - self.mean
        exponent = np.exp(-0.5 * np.dot(np.dot(x_mean.T, inv), x_mean))
        return norm_const * exponent
