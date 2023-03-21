#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    else:
        ph, pw = padding
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    conv_h = int((h + 2 * ph - kh) / sh) + 1
    conv_w = int((w + 2 * pw - kw) / sw) + 1
    output = np.zeros((m, conv_h, conv_w))
    for i in range(conv_h):
        for j in range(conv_w):
            output[:, i, j] = np.sum(padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel, axis=(1, 2))
    return output
