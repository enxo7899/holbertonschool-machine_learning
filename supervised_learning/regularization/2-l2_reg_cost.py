#!/usr/bin/env python3
"""Function l2"""


import tensorflow as tf


def l2_reg_cost(cost, lam):
    """def for t1"""
    l2_reg_cost = tf.losses.get_regularization_losses()
        return (cost + l2_reg_cost)
