#!/usr/bin/env python3
"""K means"""

import numpy as np

def kmeans(X, k, iterations=1000):
    n, d = X.shape

    # Initialize cluster centroids using a multivariate uniform distribution
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for _ in range(iterations):
        # Compute Euclidean distances between data points and cluster centroids
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)

        # Assign each data point to the nearest cluster
        clss = np.argmin(distances, axis=1)

        # Update cluster centroids
        new_C = np.empty_like(C)
        for i in range(k):
            if np.sum(clss == i) > 0:
                new_C[i] = np.mean(X[clss == i], axis=0)
            else:
                # If a cluster has no data points, reinitialize its centroid
                new_C[i] = np.random.uniform(low=min_vals, high=max_vals, size=d)

        # Check for convergence
        if np.all(C == new_C):
            return C, clss

        C = new_C

    return C, clss

# Example usage:
if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
