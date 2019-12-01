"""DCGAN model"""
import tensorflow as tf


class DCGAN():

  def __init__(self,
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
      y_dim: Dimension of dim for y. [None] #TODO what is y?
      z_dim: Dimension of dim z. [100]
      gf_dim: Dimension of gen filters in first conv layer. [64]
      df_dim: Dimension of discrim filters in first conv layer [64]
      gfc_dim: Dimension of gen units for fully connected layer. [1024]
      dfc_dim: Dimension of discrim units for fully connected layer. [1024]
      c_dim: Dimension of image color. For grayscale input, set to 1. [3]
    """
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

    # Batch normalization deals with poor initialization and helps gradient flow
    self.d_bn1 = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                    epsilon=1e-5,
                                                    scale=True)