#!/usr/bin/env python3
def summation_i_squared(n):
    if type(n) is not int or n<1:
        return None
    return int(n * (n+1) * (2*n+1) / 6)
