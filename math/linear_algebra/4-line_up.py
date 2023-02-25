#!/usr/bin/env python3
"""a function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    result = []
    if len(arr1) != len(arr2):
        return None
    else:
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
