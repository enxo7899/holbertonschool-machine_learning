#!/usr/bin/env python3
"""function to create confusion matrix"""


import numpy as np


def precision(confusion):
    """Confusion def"""
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    precision = tp / (tp + fp)
    return precision
