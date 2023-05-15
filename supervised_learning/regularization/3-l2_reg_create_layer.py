#!/usr/bin/env python3
"""Function l2"""


import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    reg = tf.keras.regularizers.l2(lambtha)
    init = tf.keras.initializers.he_normal(seed=None)
    layer = tf.keras.layers.Dense(
        units=n, 
        activation=activation, 
        kernel_initializer=init, 
        kernel_regularizer=reg)
    return layer(prev)
