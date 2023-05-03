#!/usr/bin/env python3
"""Perform optimization"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    function to rate
    """
    updated_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return updated_alpha
