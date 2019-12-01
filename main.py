"""Main for DCGAN"""
import os
import json

import tensorflow as tf
from absl import app
from absl import flags

from flags import define_flags
from flags import set_default_flags
from model import DCGAN

FLAGS = flags.FLAGS


def run(FLAGS):
  if FLAGS.dataset == 'mnist':
    dcgan = DCGAN(input_width=FLAGS.input_width,
                  input_height=FLAGS.input_height,
                  output_width=FLAGS.output_width,
                  output_height=FLAGS.output_height,
                  batch_size=FLAGS.batch_size,
                  sample_num=FLAGS.batch_size,
                  y_dim=10,
                  z_dim=FLAGS.z_dim,
                  dataset_name=FLAGS.dataset,
                  input_fname_pattern=FLAGS.input_fname_pattern,
                  crop=FLAGS.crop,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  data_dir=FLAGS.data_dir,
                  out_dir=FLAGS.out_dir,
                  max_to_keep=FLAGS.max_to_keep)
  elif FLAGS.dataset == 'celebA':
    dcgan = DCGAN(input_width=FLAGS.input_width,
                  input_height=FLAGS.input_height,
                  output_width=FLAGS.output_width,
                  output_height=FLAGS.output_height,
                  batch_size=FLAGS.batch_size,
                  sample_num=FLAGS.batch_size,
                  z_dim=FLAGS.z_dim,
                  dataset_name=FLAGS.dataset,
                  input_fname_pattern=FLAGS.input_fname_pattern,
                  crop=FLAGS.crop,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  data_dir=FLAGS.data_dir,
                  out_dir=FLAGS.out_dir,
                  max_to_keep=FLAGS.max_to_keep)
  else:
    raise Exception("No dataset exists")


def main(_):
  set_default_flags(FLAGS)
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  # Save flag setting as json.
  with open(os.path.join(FLAGS.out_dir, 'FLAGS.json'), 'w') as f:
    flags_dict = {k: FLAGS[k].value for k in FLAGS}
    json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

  run(FLAGS)


if __name__ == '__main__':
  define_flags()
  app.run(main)
