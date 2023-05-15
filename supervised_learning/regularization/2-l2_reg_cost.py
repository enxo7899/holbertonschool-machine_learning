#!/usr/bin/env python3
"""Function l2"""



import tensorflow as tf

def l2_reg_cost(cost, lam):
    l2_reg_losses = sum(tf.nn.l2_loss(w) for w in model.trainable_variables)
    return cost + lam * l2_reg_losses

    """
    for TensorFlow V1
    l2_reg_cost = tf.losses.get_regularization_losses()
        return (cost + l2_reg_cost)
    """
