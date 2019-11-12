"""Main for DCGAN"""
import os
import json

from absl import app
from absl import flags

from flags import define_flags
from utils import timestamp

FLAGS = flags.FLAGS


def run(flags_obj):
  pass


def main(_):
  # Define None flags
  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  # output folders
  if not FLAGS.out_name:
    FLAGS.out_name = '{} - {} - {}'.format(
        timestamp(),
        FLAGS.data_dir.split('/')[-1],
        FLAGS.dataset)  # penultimate folder of path
    if FLAGS.train:
      FLAGS.out_name += ' - x{}.z{}.{}.y{}.b{}'.format(FLAGS.input_width,
                                                       FLAGS.z_dim,
                                                       FLAGS.z_dist,
                                                       FLAGS.output_width,
                                                       FLAGS.batch_size)
  FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.out_name)
  FLAGS.checkpoint_dir = os.path.join(FLAGS.out_dir, FLAGS.checkpoint_dir)
  FLAGS.sample_dir = os.path.join(FLAGS.out_dir, FLAGS.sample_dir)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  with open(os.path.join(FLAGS.out_dir, 'FLAGS.json'), 'w') as f:
    flags_dict = {k: FLAGS[k].value for k in FLAGS}
    json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

  run(FLAGS)


if __name__ == '__main__':
  define_flags()
  app.run(main)