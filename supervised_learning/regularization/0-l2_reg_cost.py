#!/usr/bin/env python3
"""Function l2"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """function to calculate l2 cost"""
    l2_cost = 0
    for i in range(1, L+1):
        l2_cost += np.sum(np.square(weights['W'+str(i)]))
    l2_cost *= lambtha/(2*m)
    return cost + l2_cost
