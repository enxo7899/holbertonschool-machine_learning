#!/usr/bin/env python3
"""Function l2"""


import tensorflow as tf


def l2_reg_cost(cost):
    """def"""
    l2_loss = tf.losses.get_regularization_loss()
    total_cost = cost + l2_loss
    return total_cost
