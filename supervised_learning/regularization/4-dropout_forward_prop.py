#!/usr/bin/env python3
"""Function l2"""


import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    
    outputs = {}
    outputs["A0"] = X
    
    for index in range(L):
        
        weight = weights["W{}".format(index + 1)]
        bias = weights["b{}".format(index + 1)]
        
        z = np.matmul(weight, outputs["A{}".format(index)]) + bias
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        
        if index != (L - 1):
            A = np.tanh(z) # apply tahn activation
            A *= dropout   # apply dropout mask
            A /= keep_prob # ensure expected values do not change during training
            outputs["D{}".format(index + 1)] = dropout
        else:
            A = np.exp(z)  # apply softmax
            A /= np.sum(A, axis=0, keepdims=True)
        outputs["A{}".format(index + 1)] = A
        
    return outputs
