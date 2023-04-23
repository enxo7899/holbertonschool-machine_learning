#!/usr/bin/env python3
"""Perform binary classification with one hidden layer"""


import numpy as np


def one_hot_decode(one_hot):
    """
    function one hot
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    decoded = np.argmax(one_hot, axis=0)
    return decoded
