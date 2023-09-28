#!/usr/bin/env python3
"""
0 GP TASK 0
"""


import numpy as np


class GaussianProcess:
    """
    Gaussian Process Class
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Function Kernel
        """
        m, n = X1.shape[0], X2.shape[0]
        K = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                dist = np.linalg.norm(X1[i] - X2[j])
                K[i][j] = self.sigma_f ** 2
                K[i][j] *= np.exp(-0.5 * (dist / self.l) ** 2)

        return K


if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GaussianProcess(X_init, Y_init, l=0.6, sigma_f=2)
    print(gp.X is X_init)
    print(gp.Y is Y_init)
    print(gp.l)
    print(gp.sigma_f)
    print(gp.K.shape, gp.K)
    print(np.allclose(gp.kernel(X_init, X_init), gp.K))
