"""Download data.

Download the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
"""
import os
import argparse
import requests
import zipfile

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download dataset for DCGAN.')
parser.add_argument('datasets',
                    metavar='N',
                    type=str,
                    nargs='+',
                    choices=['celebA', 'lsun', 'mnist'],
                    help='name of dataset to download [celebA, lsun, mnist]')
DATA_DIR_PATH = './data'


def download_file_from_google_drive(id, dest):
  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={'id': id}, stream=True)
  token = get_confirm_token(response)

  if token is not None:
    params = {'id': id, 'confirm': token}
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, dest)


def save_response_content(response, dest, chunk_size=32 * 1024):
  total_size = int(response.headers.get('content-length', 0))
  with open(dest, 'wb') as f:
    for chunk in tqdm(response.iter_content(chunk_size),
                      total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc=dest):
      if chunk:  # Filter out keep-alive new chunks
        f.write(chunk)


def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None


def download_celeb_a():
  data_path = os.path.join(DATA_DIR_PATH, 'celebA')
  if os.path.exists(data_path):
    print('Found Celeb-A - skip')
    return

  filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
  save_path = os.path.join(DATA_DIR_PATH, filename)

  if os.path.exists(save_path):
    print('[*] {} already exists'.format(save_path))
  else:
    download_file_from_google_drive(drive_id, save_path)

  zip_dir = ''
  with zipfile.ZipFile(save_path) as zf:
    zip_dir = zf.namelist()[0]
    zf.extractall(DATA_DIR_PATH)
  os.remove(save_path)
  os.rename(os.path.join(DATA_DIR_PATH, zip_dir),
            os.path.join(DATA_DIR_PATH, 'celebA'))


def download_lsun():
  pass


def download_mnist():
  pass


if __name__ == '__main__':
  args = parser.parse_args()
  if not os.path.exists(DATA_DIR_PATH):
    os.mkdir(DATA_DIR_PATH)

  if 'celebA' in args.datasets:
    download_celeb_a()
  if 'lsun' in args.datasets:
    download_lsun()
  if 'mnist' in args.datasets:
    download_mnist()
  