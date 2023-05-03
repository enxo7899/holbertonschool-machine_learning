#!/usr/bin/env python3
"""Perform optimization"""


import numpy as np


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    RMS prop op
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
