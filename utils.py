"""Define utility functions"""
import time
import datetime

import numpy as np
import imageio
import cv2


def timestamp(s='%Y%m%d.%H%M%S', ts=None):
  """Get current timestamp."""
  if not ts:
    ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime(s)
  return st


def imread(path, grayscale=False):
  if grayscale:
    return imageio.imread(path).astype(np.float)
  img_bgr = cv2.imread(path)
  # Refer to: https://stackoverflow.com/questions/15072736/
  img_rgb = img_bgr[..., ::-1]
  return img_rgb.astype(np.float)
