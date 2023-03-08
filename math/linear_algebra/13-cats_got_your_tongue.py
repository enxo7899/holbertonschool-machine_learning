#!/usr/bin/env python3
"""taak 14"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """function to calculate mat"""
    return np.concatenate((mat1, mat2), axis=axis)
