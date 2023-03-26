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
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        X = data - self.mean
        self.cov = np.matmul(X, X.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        norm_const = 1.0 / (np.power((2 * np.pi),
                                     float(d) / 2) * np.power(det, 1.0 / 2))
        x_mu = x - self.mean
        result = np.exp(-0.5 * np.matmul(np.matmul(x_mu.T, inv), x_mu))
        return norm_const * result.item()
