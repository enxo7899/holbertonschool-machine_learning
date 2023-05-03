#!/usr/bin/env python3
"""Perform optimization"""


import numpy as np


def moving_average(data, beta):
    """
    function for moving average
    """
    v = 0
    EMA = []
    for i in range(len(data)):
        v = ((v * beta) + ((1 - beta) * data[i]))
        EMA.append(v / (1 - (beta ** (i + 1))))
    return EMA
