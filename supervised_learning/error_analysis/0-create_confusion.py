#!/usr/bin/env python3
"""function to create confusion matrix"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """Function to create confusion matrix"""
    return np.matmul(labels.transpose(), logits)
