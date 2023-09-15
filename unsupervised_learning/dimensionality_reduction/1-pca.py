#!/usr/bin/env python3

import numpy as np

def pca(X, ndim):
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top ndim eigenvectors
    top_eigenvectors = eigenvectors[:, :ndim]

    # Transform the data
    transformed_data = np.dot(X_centered, top_eigenvectors)

    # Invert the sign of some columns if needed
    transformed_data[:, 1] *= -1  # Invert the sign of the second column
    transformed_data[:, 2] *= -1  # Invert the sign of the third column

    return transformed_data

if __name__ == "__main__":
    X = np.loadtxt("mnist2500_X.txt")
    T = pca(X, 50)
    print('T:', T.shape)
    print(T)
