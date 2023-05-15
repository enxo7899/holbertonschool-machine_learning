#!/usr/bin/env python3
"""Function l2"""


import tensorflow as tf


def l2_reg_cost(cost):
    """asd"""
    l2_loss = tf.losses.get_regularization_losses(scope=None)
    l2_cost = tf.reduce_sum(l2_loss)
    total_cost = cost + l2_cost
    return total_cost
