#!/usr/bin/env python3
"""Function l2"""


import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob):
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(keep_prob)
    dense = tf.layers.Dense(units=n, activation=activation, kernel_initializer=init)
    dropout_layer = dropout(prev)
    output = dense(dropout_layer)
    return output
