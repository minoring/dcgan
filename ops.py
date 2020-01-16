"""Define operations"""
import tensorflow as tf


class BatchNorm(object):
  """Batch Normalization"""

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


def linear(input_,
           output_size,
           scope=None,
           stddev=0.02,
           bias_start=0.0,
           with_w=False):
  """Create weights, conduct input_ * W + b = output_size"""
  shape = input_.get_shape().as_list(0)

  with tf.variable_scope(scope or 'linear'):
    matrix = tf.get_variable('matrix', [shape[1], output_size], tf.float32,
                             tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable('bias', [output_size],
                           initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


def deconv2d(input_,
             output_shape,
             k_h=5,
             k_w=5,
             d_h=2,
             d_w=2,
             stddev=0.02,
             name='deconv2d',
             with_w=False):
  with tf.variable_scope(name):
    # Filter is shape of [height, width, out_channels, in_channels]
    w = tf.get_variable('w',
                        [k_h, k_w, output_shape[-1],
                         input_.get_shape()[-1]],
                        tf.float32,
                        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_,
                                    w,
                                    output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]],
                             initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    return deconv


def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat(
      [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
