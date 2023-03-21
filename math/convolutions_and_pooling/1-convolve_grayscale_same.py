#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""


import numpy as np
def convolve_grayscale_same(images, kernel):
    """
    Performs a valid convolution on grayscale images
    """


m, h, w = images.shape
    kh, kw = kernel.shape
    ph = max(kh - 1, 0)
    pw = max(kw - 1, 0)
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),                         mode='constant', constant_values=0)
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel,                              axis=(1,2))
    return output
