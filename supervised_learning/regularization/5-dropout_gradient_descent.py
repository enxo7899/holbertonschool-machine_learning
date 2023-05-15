#!/usr/bin/env python3
"""Function l2"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):

    m = Y.shape[1]
    back = {}
    
    for index in range(L, 0, -1):
        A = cache["A{}".format(index - 1)]
        if index == L:
            # if last layer, calculate dz value as A - Y
            back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
            dz = back["dz{}".format(index)]

        else:
            # retreive previous dz and current activation
            dz_prev = back["dz{}".format(index + 1)]
            A_current = cache["A{}".format(index)]
            
            # calculate current dz
            back["dz{}".format(index)] = (
                np.matmul(W_prev.transpose(), dz_prev) *
                (A_current * (1 - A_current)))
            
            # multiply by dropout mask and divide by keep probability
            dz = back["dz{}".format(index)]
            dz *= cache["D{}".format(index)]
            dz /= keep_prob

        # calculate weights and bias
        dW = (1 / m) * (np.matmul(dz, A.transpose()))
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        
        # store for next interation
        W_prev = weights["W{}".format(index)]
        
        # modify weights and bias
        weights["W{}".format(index)] = (
            weights["W{}".format(index)] - (alpha * dW))
        weights["b{}".format(index)] = (
            weights["b{}".format(index)] - (alpha * db))
