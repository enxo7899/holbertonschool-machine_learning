#!/usr/bin/env python3
"""
Defines the class GRUCell that represents a gated recurrent unit
"""


import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit

    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))

        """
        The first dimension, h + i, determines the total number of input
        units to the update gate. 
        This includes the hidden state dimension
        (h) and the input dimension (i). 

        The input dimension i represents the size
        of the input at each time step, 
        while the hidden state dimension h represents the
        size of the previous hidden state.
        """
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))

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

    def sigmoid(self, x):
        """
        Performs the sigmoid function
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        """

        """
        perform the computation of the update gate (z_gate) and
        reset gate (r_gate) for the GRU cell.
        """
        # concatenating the previous hidden state and hidden input
        concatenation1 = np.concatenate((h_prev, x_t), axis=1)
        z_gate = self.sigmoid(np.matmul(concatenation1, self.Wz) + self.bz)
        r_gate = self.sigmoid(np.matmul(concatenation1, self.Wr) + self.br)

        """
        perform the computation of the next hidden state 
        based on the reset gate, the previous hidden state, 
        the current input, and the update gate.
        """
        concatenation2 = np.concatenate((r_gate * h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation2, self.Wh) + self.bh)
        # element-wise multiplication between the intermediate hidden state h_next 
        # and the update gate z_gate; controls information (0, 1)
        h_next *= z_gate
        #  the complement of the update gate, how much info to discard
        h_next += (1 - z_gate) * h_prev

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
