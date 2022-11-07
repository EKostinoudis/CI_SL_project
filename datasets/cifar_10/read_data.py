#!/usr/bin/env python3
from pathlib import Path
from os.path import exists
import numpy as np

def extract_tar_gz_if_needed(file):
    import tarfile
    file = tarfile.open(file)
    file_names = file.getnames()
    file_names = [Path(__file__).parent.joinpath(file) for file in file_names]
    if any([not exists(file) for file in file_names]):
        file.extractall()
    file.close()

def download_if_not_exists(file):
    if exists(file):
        return
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    import requests
    response = requests.get(url)
    open(file, "wb").write(response.content)

def ensure_data_exists():
    path_data = Path(__file__).with_name('cifar-10-python.tar.gz')
    download_if_not_exists(path_data)
    extract_tar_gz_if_needed(path_data)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_train_batch(batch_num):
    ensure_data_exists()

    if 1 <= batch_num <= 5:
        return unpickle(f'cifar-10-batches-py/data_batch_{batch_num}')
    else:
        raise Exception(f"Incorrect batch number {batch_num}")

def load_test_batch():
    ensure_data_exists()
    return unpickle('cifar-10-batches-py/test_batch')

