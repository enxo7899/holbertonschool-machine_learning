#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function for RBG images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    out_h = int((h + 2*ph - kh) / sh) + 1
    out_w = int((w + 2*pw - kw) / sw) + 1
    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel, axis=(1, 2))
    return output
