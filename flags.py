"""Define flags for DCGAN"""

import numpy as np
from absl import flags


def define_flags():
  flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
  flags.DEFINE_float("learning_rate", 0.0002,
                     "Learning rate of for adam [0.0002]")
  flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
  flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
  flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
  flags.DEFINE_integer(
      "input_height", 108,
      "The size of image to use (will be center cropped). [108]")
  flags.DEFINE_integer(
      "input_width", None,
      "The size of image to use (will be center cropped). If None, same value as input_height [None]"
  )
  flags.DEFINE_integer("output_height", 64,
                       "The size of the output images to produce [64]")
  flags.DEFINE_integer(
      "output_width", None,
      "The size of the output images to produce. If None, same value as output_height [None]"
  )
  flags.DEFINE_string("dataset", "celebA",
                      "The name of dataset [celebA, mnist, lsun]")
  flags.DEFINE_string("input_fname_pattern", "*.jpg",
                      "Glob pattern of filename of input images [*]")
  flags.DEFINE_string("data_dir", "./data",
                      "path to datasets [e.g. $HOME/data]")
  flags.DEFINE_string("out_dir", "./out",
                      "Root directory for outputs [e.g. $HOME/out]")
  flags.DEFINE_string(
      "out_name", "",
      "Folder (under out_root_dir) for all outputs. Generated automatically if left blank []"
  )
  flags.DEFINE_string(
      "checkpoint_dir", "checkpoint",
      "Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
  flags.DEFINE_string(
      "sample_dir", "samples",
      "Folder (under out_root_dir/out_name) to save samples [samples]")
  flags.DEFINE_boolean("train", False,
                       "True for training, False for testing [False]")
  flags.DEFINE_boolean("crop", False,
                       "True for training, False for testing [False]")
  flags.DEFINE_boolean("visualize", False,
                       "True for visualizing, False for nothing [False]")
  flags.DEFINE_boolean("export", False,
                       "True for exporting with new batch size")
  flags.DEFINE_boolean("freeze", False,
                       "True for exporting with new batch size")
  flags.DEFINE_integer("max_to_keep", 1,
                       "maximum number of checkpoints to keep")
  flags.DEFINE_integer("sample_freq", 200, "sample every this many iterations")
  flags.DEFINE_integer("ckpt_freq", 200,
                       "save checkpoint every this many iterations")
  flags.DEFINE_integer("z_dim", 100, "dimensions of z")
  flags.DEFINE_string("z_dist", "uniform_signed",
                      "'normal01' or 'uniform_unsigned' or uniform_signed")
  flags.DEFINE_boolean("G_img_sum", False,
                       "Save generator image summaries in log")
