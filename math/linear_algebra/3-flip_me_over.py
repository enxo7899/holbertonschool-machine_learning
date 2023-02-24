#!/usr/bin/env python3
"""a function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    result = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        result.append(row)
    return result
