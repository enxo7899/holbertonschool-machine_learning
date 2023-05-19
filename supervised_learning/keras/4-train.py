#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """function to train model"""
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
