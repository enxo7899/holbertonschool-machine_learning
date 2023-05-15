#!/usr/bin/env python3
"""Function l2"""


import numpy as np
import tensorflow as tf

def dropout_forward_prop(X, weights, L, keep_prob):
    cache = {'A0': X}
    dropout_masks = {}
    
    for i in range(1, L + 1):
        A_prev = cache[f'A{i - 1}']
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        
        Z = tf.matmul(W, A_prev) + b
        if i != L:
            A = tf.nn.tanh(Z)
            D = tf.random_uniform(tf.shape(A), minval=0, maxval=1) < keep_prob
            A = tf.multiply(A, tf.cast(D, dtype=tf.float32))
            A /= keep_prob
            dropout_masks[f'D{i}'] = D
        else:
            A = tf.nn.softmax(Z)
        
        cache[f'A{i}'] = A
    
    return {**cache, **dropout_masks}
