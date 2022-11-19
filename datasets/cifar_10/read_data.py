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
        file = Path(__file__).parent.joinpath(f'cifar-10-batches-py/data_batch_{batch_num}')
        return unpickle(file)
    else:
        raise Exception(f"Incorrect batch number {batch_num}")

def load_batch_meta():
    ensure_data_exists()
    file = Path(__file__).parent.joinpath('cifar-10-batches-py/batches.meta')
    return unpickle(file)

def load_test_batch():
    ensure_data_exists()
    file = Path(__file__).parent.joinpath('cifar-10-batches-py/test_batch')
    return unpickle(file)

def load_train_data():
    labels = []
    data = None
    # load data for all the batches
    for i in range(1,6):
        batch_data = load_train_batch(i)
        # data = np.vstack(data, batch_data[b'data'])
        data = np.concatenate((data, batch_data[b'data']), axis=0) if data is not None else batch_data[b'data']
        labels.extend(batch_data[b'labels'])
    return data, np.array(labels, dtype=np.uint8)

def load_test_data():
    batch_data = load_test_batch()
    return batch_data[b'data'], np.array(batch_data[b'labels'], dtype=np.uint8)

def load_label_names():
    batch_data = load_batch_meta()
    return list(x.decode() for x in batch_data[b'label_names'])
