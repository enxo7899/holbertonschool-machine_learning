#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """function to save weight"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """function to load weight"""
    network.load_weights(filename)
    return None
