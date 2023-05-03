#!/usr/bin/env python3
"""Perform optimization"""


import tensorflow as tf


def batch_norm(Z, gamma, beta, epsilon):
    """
    batch norm
    """
    m, n = Z.shape
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    Z_norm_scaled  = gamma * Z_norm + beta
    return Z_norm_scaled 
