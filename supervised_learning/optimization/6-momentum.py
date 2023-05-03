#!/usr/bin/env python3
"""Perform optimization"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    function to create momentum
    """
    optimizer = tf.train.MomentumOptimizer(alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op
