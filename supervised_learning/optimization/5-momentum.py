#!/usr/bin/env python3
"""Perform optimization"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    function to update variable momentum
    """
    dW_prev = (beta1 * v) + ((1 - beta1) * grad)
    var -= (alpha * dW_prev)
    return var, dW_prev
