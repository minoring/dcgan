"""Implement DCGAN model"""
import os
from glob import glob

import tensorflow as tf
import numpy as np

from ops import BatchNorm
from utils import imread


class DCGAN(object):

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
      data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
      self.data = glob(data_path)
      if len(self.data) == 0:
        raise Exception("[!] No data found in '" + data_path + "'")

      if len(self.data) < self.batch_size:
        raise Exception('[!] Entire dataset size is less then the configured batch size')

      np.random.shuffle(self.data)
      imread_img = imread(self.data[0])
      if len(imread_img.shape) >= 3: # Check if image is a non-grayscale image
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

      self.gray_scale = self.c_dim == 1

      self.build_model()
