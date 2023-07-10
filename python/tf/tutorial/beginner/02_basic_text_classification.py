import os
import re
import shutil
import string

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses


print(tf.__version__)


def main():
    datset = download_data()


def download_data():
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dataset = tf.keras.utils.get_file(
        'aclImdb_v1', url, untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    print('dataset dir:', os.listdir(dataset_dir))
    train_dir = os.path.join(dataset_dir, 'train')
    print('train dir:', os.listdir(train_dir))
    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    with open(sample_file) as f:
        print(f.read())
