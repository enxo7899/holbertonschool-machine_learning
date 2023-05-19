#!/usr/bin/env python3
"""A function that builds a neural network with the Keras library"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """function one hot"""
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
