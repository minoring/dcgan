"""Define utility functions"""
import time
import datetime

def timestamp(s='%Y%m%d.%H%M%S', ts=None):
  """Get current timestamp."""
  if not ts:
    ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime(s)
  return st