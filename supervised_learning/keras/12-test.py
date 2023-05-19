#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make predictions using the network model"""
    predictions = network.predict(data)
    if verbose:
        print(predictions)
    return predictions
