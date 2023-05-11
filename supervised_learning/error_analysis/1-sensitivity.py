#!/usr/bin/env python3
"""function to create confusion matrix"""


import numpy as np


def sensitivity(confusion):
    """Sensitiviy function"""
    tp = np.diag(confusion)
    fn = np.sum(confusion, axis=1) - tp
    tpr = tp / (tp + fn)
    return tpr
