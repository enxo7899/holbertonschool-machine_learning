#!/usr/bin/env python3
"""A function that builds a neural network with the Keras library"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """function to optimize model"""
    network.compile(optimizer=K.optimizers.Adam(lr=alpha,
                                                beta_1=beta1,
                                                beta_2=beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    return None
