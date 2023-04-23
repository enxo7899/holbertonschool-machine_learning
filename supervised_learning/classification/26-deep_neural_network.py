#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Constructor method

        nx: int: number of input features
        layers: list: number of nodes in each layer of the network
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for i in layers:
            if type(i) is not int or i < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if i == 0:
                self.__weights['W1'] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W{}'.format(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        Getter method for __L private attribute
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method for __cache private attribute
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method for __weights private attribute
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        X: numpy.ndarray: shape (nx, m): contains the input data
        nx: int: number of input features
        m: int: number of examples

        Returns: output of the neural network and cache, respectively
        """

        self.__cache['A0'] = X
        for i in range(self.__L):
            W_key = 'W{}'.format(i + 1)
            b_key = 'b{}'.format(i + 1)
            A_key_prev = 'A{}'.format(i)
            A_key_forw = 'A{}'.format(i + 1)
            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Y: numpy.ndarray: shape (1, m): contains the correct labels
                              for the input data
        A: numpy.ndarray: shape (1, m): containing the activated output
                              of the neuron for each example

        Returns: cost
        """

        m = Y.shape[1]
        cost = -(1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
