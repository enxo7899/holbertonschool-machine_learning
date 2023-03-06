#!/usr/bin/env python3
import numpy as np
def np_slice(matrix, axes={}):
    for axis, s in axes.items():
        matrix = np.take(matrix, indices=range(*s), axis=axis)
    return matrix
