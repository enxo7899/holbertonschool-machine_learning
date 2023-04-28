#!/usr/bin/env python3
"""
Create the training operation for the neural network
"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Create the training operation for the network
    """
    gradient_descent = tf.train.GradientDescentOptimizer(alpha)
    return (gradient_descent.minimize(loss))
