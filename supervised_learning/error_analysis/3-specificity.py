#!/usr/bin/env python3
"""function to create confusion matrix"""


import numpy as np


def specificity(confusion):
    """Calculate the true negative rate (specificity) for each class"""
    tn = (
        np.sum(confusion) - np.sum(confusion, axis=0) -
        np.sum(confusion, axis=1) + np.diag(confusion)
    )
    fp = np.sum(confusion, axis=0) - np.diag(confusion)
    tnr = tn / (tn + fp)
    return tnr
