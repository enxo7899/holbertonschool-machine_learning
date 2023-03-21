#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    A function to convlovle rbg images 
    """
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    oh = int((h + 2 * ph - kh) / sh + 1)
    ow = int((w + 2 * pw - kw) / sw + 1)
    output = np.zeros((m, oh, ow, nc))
    for i in range(oh):
        for j in range(ow):
            output[:, i, j, :] = np.sum(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :, np.newaxis] * kernels, axis=(1, 2, 3))
    return output
