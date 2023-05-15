#!/usr/bin/env python3
"""Function l2"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """function for l2"""
    initializer = (
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    )
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = (
        tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer)
    )
    output = layer(prev)
    return output
