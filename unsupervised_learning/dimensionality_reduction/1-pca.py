#!/usr/bin/env python3
import numpy as np

def pca(X, ndim):
    # Calculate the mean of each feature:
    mean_vect = np.mean(X, axis=0)

    # Subtract the mean from each feature:
    centered_X = X - mean_vect

    # Calculate the covariance matrix:
    cov_matrix = np.cov(centered_X.T)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix:
    eig_vals, eig_vects = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues by their magnitude:
    eig_pairs = [(np.abs(eig_vals[i]), eig_vects[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Choose the top ndim eigenvectors:
    eig_vects_ndim = np.array([eig_pairs[i][1] for i in range(ndim)])

    # Project the data onto the top ndim eigenvectors:
    T = np.dot(centered_X, eig_vects_ndim.T)

    return T
