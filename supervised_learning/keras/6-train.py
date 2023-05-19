#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """function to train model"""
    if early_stopping and validation_data:
        callback = []
        callback.append(
            K.callbacks.EarlyStopping(monitor='loss', patience=patience))
    else:
        callback = None

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
