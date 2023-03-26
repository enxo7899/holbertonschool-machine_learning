#!/usr/bin/env python3
"""Likelihood"""

import numpy as np


def likelihood(x, n, P):
    """
    A function to calculate likehood
    """
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if x > n:
        raise ValueError("x cannot be greater than n")
    binom_coeff = np.math.comb(n, x)
    likelihoods = binom_coeff * np.power(P, x) * np.power(1 - P, n - x)
    return likelihoods
