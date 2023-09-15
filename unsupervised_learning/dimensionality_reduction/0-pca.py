#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """
    Perform PCA on a dataset.

    Args:
        X (numpy.ndarray): The input dataset with shape (n, d).
        var (float): The fraction of variance to maintain (default is 0.95).

    Returns:
        numpy.ndarray: The weights matrix W with shape (d, nd) where nd is
                      the new dimensionality of the transformed X.
    """
    # Calculate the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the cumulative explained variance
    explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Determine the number of dimensions to keep
    nd = np.argmax(explained_variance_ratio >= var) + 1

    # Select the top nd eigenvectors as the transformation matrix
    W = eigenvectors[:, :nd]

    return W

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.normal(size=50)
    b = np.random.normal(size=50)
    c = np.random.normal(size=50)
    d = 2 * a
    e = -5 * b
    f = 10 * c

    X = np.array([a, b, c, d, e, f]).T
    m = X.shape[0]
    X_m = X - np.mean(X, axis=0)
    W = pca(X_m)
    T = np.matmul(X_m, W)
    print(T)
    X_t = np.matmul(T, W.T)
    print(np.sum(np.square(X_m - X_t)) / m)
