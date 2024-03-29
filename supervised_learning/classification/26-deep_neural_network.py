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

    @property
    def L(self):
        return (self.__L)

    @property
    def cache(self):
        return (self.__cache)

    @property
    def weights(self):
        return (self.__weights)

    def forward_prop(self, X):

        self.__cache["A0"] = X

        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]

            z = np.matmul(W, self.cache["A{}".format(index)]) + b
            A = 1 / (1 + (np.exp(-z)))

            self.__cache["A{}".format(index + 1)] = A

        return (A, self.cache)

    def cost(self, Y, A):
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)

    def evaluate(self, X, Y):
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):

        m = Y.shape[1]
        back = {}

        for index in range(self.L, 0, -1):

            A = cache["A{}".format(index - 1)]

            if index == self.L:
                back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
            else:
                dz_prev = back["dz{}".format(index + 1)]
                A_current = cache["A{}".format(index)]
                back["dz{}".format(index)] = (
                    np.matmul(W_prev.transpose(), dz_prev) *
                    (A_current * (1 - A_current)))

            dz = back["dz{}".format(index)]
            A_prev = cache["A{}".format(index - 1)]
            W = self.weights["W{}".format(index)]
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dW = (1 / m) * np.matmul(dz, A_prev.T)
            W_prev = self.weights["W{}".format(index+1)]
            self.weights["W{}".format(index)] -= alpha * dW
            self.weights["b{}".format(index)] -= alpha * db
     
    
    def get_max_profit(prices):
        if len(prices) < 2:
            raise ValueError("Getting a profit requires at least 2 prices")

        min_price = prices[0]
        max_profit = prices[1] - prices[0]

        for i in range(1, len(prices)):
            current_price = prices[i]
            potential_profit = current_price - min_price
            max_profit = max(max_profit, potential_profit)
            min_price = min(min_price, current_price)

        return max_profit
