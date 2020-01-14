"""Define operations"""
import tensorflow as tf


class BatchNorm(object):

  def __init__(self, epsilon=1e-8, momentum=0.9, name='batch_norm'):
    with tf.compat.v1.variable_scope(name):
      self.epsilon = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                              epsilon=self.epsilon,
                                              scale=True,
                                              name=self.name)