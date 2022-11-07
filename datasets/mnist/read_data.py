#!/usr/bin/env python3
import gzip
import numpy as np
from pathlib import Path
from os.path import exists, basename

def read_image_data(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        magic = int.from_bytes(data[0:4], 'big', signed=True)
        if magic != 2051:
            raise Exception(f"Wrong magic in file {filename}")

        num_of_images = int.from_bytes(data[4:8], 'big', signed=True)
        rows = int.from_bytes(data[8:12], 'big', signed=True)
        cols = int.from_bytes(data[12:16], 'big', signed=True)

        images = np.zeros((num_of_images, rows, cols), dtype=np.uint8)
        images[:,:,:] = np.array([int(d) for d in data[16:]]).reshape(num_of_images, rows, cols)
        return images

def read_label_data(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        magic = int.from_bytes(data[0:4], 'big', signed=True)
        if magic != 2049:
            raise Exception(f"Wrong magic in file {filename}")

        num_of_labels = int.from_bytes(data[4:8], 'big', signed=True)

        labels = np.zeros(num_of_labels, dtype=np.uint8)
        labels[:] = np.array([int(d) for d in data[8:]]).reshape(labels.shape)
        return labels

def download_if_not_exists(file):
    if exists(file):
        return
    url = f'http://yann.lecun.com/exdb/mnist/{basename(file)}'
    import requests
    response = requests.get(url)
    open(file, "wb").write(response.content)

def load_train_data():
    images_filename = 'train-images-idx3-ubyte.gz'
    labels_filename = 'train-labels-idx1-ubyte.gz'

    path_images = Path(__file__).with_name(images_filename)
    path_labels = Path(__file__).with_name(labels_filename)

    download_if_not_exists(path_images)
    download_if_not_exists(path_labels)

    images = read_image_data(path_images)
    labels = read_label_data(path_labels)
    return (images, labels)

def load_test_data():
    images_filename = 't10k-images-idx3-ubyte.gz'
    labels_filename = 't10k-labels-idx1-ubyte.gz'

    path_images = Path(__file__).with_name(images_filename)
    path_labels = Path(__file__).with_name(labels_filename)

    download_if_not_exists(path_images)
    download_if_not_exists(path_labels)

    images = read_image_data(path_images)
    labels = read_label_data(path_labels)
    return (images, labels)

def load_all_data():
    train_image, train_labels = load_train_data()
    test_image, test_labels = load_test_data()
    return (np.concatenate((train_image, test_image), axis=0), np.concatenate((train_labels, test_labels), axis=0))
