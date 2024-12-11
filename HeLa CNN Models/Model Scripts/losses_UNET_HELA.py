# This module defines the loss functions for each of the models used in training and validation.
# The Encoder, Generator, and Discriminator models each have their own loss funtion based on what the
# models are actually trying to accomplish.

# Importing packages

import tensorflow as tf

# Reconstruction loss function

def ReconstructionLoss(target, gen_output):

    epsilon_ = 1e-07
    bce = target * tf.math.log(gen_output + epsilon_)
    bce += (1 - target) * tf.math.log(1 - gen_output + epsilon_)
    loss = tf.reduce_sum(-bce, axis=[1,2,3])
    loss = tf.reduce_mean(loss)

    return loss
