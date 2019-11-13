"""Define operations"""
import tensorflow as tf


class batch_norm():

  def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
    with tf.compat.v1.variable_scope(name): # TODO 이거 왜하는거지 Tensor만드는것도 아닌데.
      self.epsilon = epsilon
      self.momentum = momentum
      self.name = name
  
  def __call__(self, x, train=True):
    
