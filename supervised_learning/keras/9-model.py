#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K


def save_model(network, filename):
    """function to save model"""
    network.save(filename)
    return None


def load_model(filename):
    """function to load model"""
    model = K.models.load_model(filename)
    return model
