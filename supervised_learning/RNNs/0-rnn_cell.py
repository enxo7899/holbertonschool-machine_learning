#!/usr/bin/env python3
"""
Defines the class RNNCell that represents a cell of a simple RNN
"""


import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN

    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Performs the softmax function
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)

        return softmax

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        concatenation = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
