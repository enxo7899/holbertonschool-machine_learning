#!/usr/bin/env python3
"""
Defines function that performs forward propagation for simple RNN
"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for simple RNN
    """
    t, m, i = X.shape
    m, h = h_0.shape

    """
    initializes a matrix H with zeros
    t + 1: time steps plus a row for the initial hidden state. 
    m: columns, i.e., number of samples,
    h: depth, i.e., the hidden state dimension.
    """
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    """
    forward propagation for each time step of the simple RNN
    updating the hidden states and collecting the outputs
    """
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        if step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))

    """
    the output matrix is organized based on the dimensions of time steps and samples
    allowing for easier interpretation and further processing of the RNN outputs.
    determines the size of the last dimension of the matrix Y
    representing the shape of the output at each time step.
    """
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)

    return (H, Y)
