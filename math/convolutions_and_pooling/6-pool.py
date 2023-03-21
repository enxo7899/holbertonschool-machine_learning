#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function to pool images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    oh = int((h - kh) / sh + 1)
    ow = int((w - kw) / sw + 1)
    pooled_images = np.zeros((m, oh, ow, c))
    for i in range(m):
        for j in range(c):
            if mode == 'max':
                pooled_images[i, :, :, j] = np.max([
                    [
                        images[i, r * sh:r * sh + kh, c * sw:c * sw + kw, j]
                        for c in range(ow)
                    ]
                    for r in range(oh)
                ], axis=(1, 2))
            elif mode == 'avg':
                pooled_images[i, :, :, j] = np.mean([
                    [
                        images[i, r * sh:r * sh + kh, c * sw:c * sw + kw, j]
                        for c in range(ow)
                    ]
                    for r in range(oh)
                ], axis=(1, 2))
    return pooled_images
