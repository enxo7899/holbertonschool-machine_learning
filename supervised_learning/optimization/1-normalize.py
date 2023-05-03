#!/usr/bin/env python3
"""Perform optimization"""


import numpy as np

def normalize(X, m, s):
    """
    Normalize X by subtracting the mean and dividing by the standard deviation
    """
    X_norm = (X - m) / s
    return X_norm
