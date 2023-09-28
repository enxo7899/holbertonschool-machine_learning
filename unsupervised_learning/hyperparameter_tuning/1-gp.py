#!/usr/bin/env python3
"""
Hyperparameter Tunning
"""


import numpy as np

class GaussianProcess:
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        # Existing constructor code...

    def kernel(self, X1, X2):
        # Existing kernel calculation code...

    def predict(self, X_s):
        """
        function
        """
        s = X_s.shape[0]
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        # Calculate the predictive mean
        mu = K_s.T.dot(np.linalg.inv(self.K)).dot(self.Y).flatten()

        # Calculate the predictive variance
        sigma = np.diag(K_ss - K_s.T.dot(np.linalg.inv(self.K)).dot(K_s))

        return mu, sigma
