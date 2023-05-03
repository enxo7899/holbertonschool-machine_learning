#!/usr/bin/env python3
"""Perform optimization"""


import tensorflow as tf


import tensorflow as tf

def model(Data_train, Data_valid, layers, activations, 
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, 
          decay_rate=1, batch_size=32, epochs=5, save_path='./model.ckpt'):
    """
    def train
    """
    X_train, y_train = Data_train
    X_valid, y_valid = Data_valid   
    input_shape = X_train.shape[1:]
    model = tf.keras.models.Sequential()   
    # Add the layers to the model
    for i, layer_size in enumerate(layers):
        activation = activations[i]
        if i == 0:
            model.add(tf.keras.layers.Dense(layer_size, input_shape=input_shape, activation=activation))
        else:
            model.add(tf.keras.layers.Dense(layer_size, activation=activation))       
        model.add(tf.keras.layers.BatchNormalization())   
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)   
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()   
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: alpha / (1 + decay_rate * epoch))
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_best_only=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[lr_scheduler, model_checkpoint])
    return save_path
