import os
import re
import shutil
import string

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses


print(tf.__version__)


BATCH = 32
SEED = 42


def main():
    raw_dataset = download_data()


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
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=BATCH,
        validation_split=0.2,
        subset='training',
        seed=SEED)
    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            print('Review:', text_batch.numpy()[i])
            print('Label:', label_batch.numpy()[i]) # 0: neg; 1: pos
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=BATCH,
        validation_split=0.2,
        subset='validation',
        seed=SEED)
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test', batch_size=BATCH)
    return raw_train_ds, raw_val_ds, raw_test_ds
