#!/usr/bin/env python3
"""Defines a single neuron performing binary classfication"""

import numpy as np


class Neuron:
    """
    Single Nueron performing binary classification
    """
    # TASK 1
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return (self.__W)

    @property
    def b(self):
        return (self.__b)

    @property
    def A(self):
        return (self.__A)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression function:
        loss = -((Y * log(A)) + ((1 - Y) * log(1 - A)))
        logistic regression cost function:
        cost = (1 / m) * sum of loss function for all m example
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)

    def evaluate(self, X, Y):
        """
        evaluates the neuron's predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        calculates one pass of gradient descent on the neuron
        derivative of loss function with respect to A:
        dA = (-Y / A) + ((1 - Y) / (1 - A))
        derivative of A with respect to z:
        dz = A * (1 - A)
        combining two above with chain rule,
        derivative of loss function with respect to z:
        dz = A - Y
        using chain rule with above derivative,
        derivative of loss function with respect to __W:
        d__W = Xdz
        derivative of loss function with respect to __b:
        d__b = dz
        one-step of gradient descent updates the attributes with the following:
        __W = __W - (alpha * d__W)
        __b = __b - (alpha * d__b)
        """
        m = Y.shape[1]
        dz = (A - Y)
        d__W = (1 / m) * (np.matmul(X, dz.transpose()).transpose())
        d__b = (1 / m) * (np.sum(dz))
        self.__W = self.W - (alpha * d__W)
        self.__b = self.b - (alpha * d__b)
