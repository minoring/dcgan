"""Implement DCGAN model"""
import os
import math
from glob import glob

import tensorflow as tf
import numpy as np

from ops import BatchNorm
from ops import linear
from ops import deconv2d
from utils import imread


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
  """Generative Adversarial Networks"""

  def __init__(self,
               sess,
               input_height=108,
               input_width=108,
               crop=True,
               batch_size=64,
               sample_num=64,
               output_height=64,
               output_width=64,
               y_dim=None,
               z_dim=100,
               gf_dim=64,
               df_dim=64,
               gfc_dim=1024,
               dfc_dim=1024,
               c_dim=3,
               dataset_name='default',
               max_to_keep=1,
               input_fname_pattern='*.jpg',
               checkpoint_dir='ckpts',
               sample_dir='samples',
               out_dir='./out',
               data_dir='./data'):
    """DCGAN model

    Args:
      sess: TensorFlow session
      batch_size: The size of batch.
      y_dim: (optional) Dimension of dim for y. [None] #TODO what is y?
      z_dim: (optional) Dimension of dim z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer [64]
      gfc_dim: (optional) Dimension of gen units for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.input_height = input_height
    self.input_width = input_width
    self.crop = crop
    self.batch_size = batch_size
    self.sample_num = sample_num
    self.output_height = output_height
    self.output_width = output_width
    self.y_dim = y_dim
    self.z_dim = z_dim
    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim
    self.c_dim = c_dim
    self.dataset_name = dataset_name
    self.max_to_keep = max_to_keep
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.out_dir = out_dir
    self.data_dir = data_dir
    self.y = None

    # Batch normalization deals with poor initialization and helps gradient flow
    self.d_bn1 = BatchNorm(name='d_bn1')
    self.d_bn2 = BatchNorm(name='d_bn2')

    if self.y_dim is None:
      self.d_bn3 = BatchNorm(name='d_bn3')

    self.g_bn0 = BatchNorm(name='g_bn0')
    self.g_bn1 = BatchNorm(name='g_bn1')
    self.g_bn2 = BatchNorm(name='g_bn2')

    if self.y_dim is None:
      self.g_bn3 = BatchNorm(name='g_bn3')

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X.shape[-1]
    else:
      data_path = os.path.join(self.data_dir, self.dataset_name,
                               self.input_fname_pattern)
      self.data = glob(data_path)
      if len(self.data) == 0:
        raise Exception("[!] No data found in '" + data_path + "'")

      if len(self.data) < self.batch_size:
        raise Exception(
            '[!] Entire dataset size is less then the configured batch size')

      np.random.shuffle(self.data)
      imread_img = imread(self.data[0])
      if len(imread_img.shape) >= 3:  # Check if image is a non-grayscale image
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

      self.gray_scale = self.c_dim == 1

      self.build_model()

  def build_molde(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim],
                              name='y')

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims,
                                 name='real_images')
    inputs = self.inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_summ = tf.histogram_summary('z', self.z)

    self.G = self.generator(self.z, self.y)

  def generator(self, z, y=None):
    with tf.variable_scope('generator') as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # Project `z` and reshape.
        self.z_, self.h0_w, self.h0_b = linear(z,
                                               self.gf_dim * 8 * s_h16 * s_w16,
                                               'g_h0_lin',
                                               with_w=True)
        self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
            name='g_h1',
            with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
            name='g_h2',
            with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
            name='g_h3',
            with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim],
            name='g_h4',
            with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
        s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = tf.concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(
            self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(
            self.g_bn2(
                deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2],
                         name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            decond2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
