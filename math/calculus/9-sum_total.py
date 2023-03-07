#!/usr/bin/env python3
""" A function that calculates the sums of squares"""


def summation_i_squared(n):
    """ Summation of squared numbers"""
    if type(n) is not int or n < 1:
        return None
    return int(n * (n + 1) * (2*n + 1) / 6)
