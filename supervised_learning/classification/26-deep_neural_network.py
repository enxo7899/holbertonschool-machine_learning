#!/usr/bin/env python3
"""Perform binary classification with one hidden layer"""


import numpy as np


import pickle


class DeepNeuralNetwork:
    """
    class that represents a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):

            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        """
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None
