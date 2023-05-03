#!/usr/bin/env python3
"""Perform optimization"""


import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
    bactch norm layer
    """
    k_init = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    dense_layer = tf.keras.layers.Dense(units=n, kernel_initializer=k_init, use_bias=False)
    Z = dense_layer(prev)
    beta = tf.Variable(initial_value=tf.zeros((1, n)), trainable=True)
    gamma = tf.Variable(initial_value=tf.ones((1, n)), trainable=True)
    epsilon = 1e-8
    mean, variance = tf.nn.moments(Z, axes=0, keepdims=True)
    Z_norm = (Z - mean) / tf.sqrt(variance + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return activation(Z_tilde)
