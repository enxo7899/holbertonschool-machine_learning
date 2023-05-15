#!/usr/bin/env python3
"""Function l2"""


import numpy as np
import tensorflow as tf

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    m = Y.shape[1]    
    dZ = cache['A' + str(L)] - Y
    dW = (1 / m) * tf.matmul(dZ, tf.transpose(cache['A' + str(L - 1)]))
    db = (1 / m) * tf.reduce_sum(dZ, axis=1, keepdims=True)
    weights['W' + str(L)] -= alpha * dW
    weights['b' + str(L)] -= alpha * db  
    for i in range(L - 1, 0, -1):
        dA = tf.matmul(tf.transpose(weights['W' + str(i + 1)]), dZ)
        dA *= cache['D' + str(i)] / keep_prob
        dZ = dA * (1 - tf.square(tf.tanh(cache['A' + str(i)])))
        dW = (1 / m) * tf.matmul(dZ, tf.transpose(cache['A' + str(i - 1)]))
        db = (1 / m) * tf.reduce_sum(dZ, axis=1, keepdims=True)
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
