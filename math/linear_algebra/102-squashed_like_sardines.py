#!/usr/bin/env python3
def cat_matrices(mat1, mat2, axis=0):
    import numpy as np
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        if axis == 0:
            return mat1 + mat2
        elif axis == 1:
            return [mat1[i] + mat2[i] for i in range(len(mat1))]
    np_mat1 = np.array(mat1)
    np_mat2 = np.array(mat2)
    try:
        np_result = np.concatenate((np_mat1, np_mat2), axis=axis)
    except ValueError:
        return None
    return np_result.tolist()
