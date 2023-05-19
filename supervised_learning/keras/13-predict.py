#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """function to predict"""
    prediction = network.predict(x=data,
                                 verbose=verbose)
    return prediction
