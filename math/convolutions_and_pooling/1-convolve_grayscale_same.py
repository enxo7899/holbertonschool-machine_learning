#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a valid convolution on grayscale images
    """
    m, height, width = images.shape
    kh, kw = kernel.shape
    if (kh % 2) == 1:
        ph = (kh - 1) // 2
    else:
        ph = kh // 2
    if (kw % 2) == 1:
        pw = (kw - 1) // 2
    else:
        pw = kw // 2
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    convoluted = np.zeros((m, height, width))
    for h in range(height):
        for w in range(width):
            output = np.sum(images[:, h: h + kh, w: w +
                                   kw] * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
