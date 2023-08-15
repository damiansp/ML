import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as ds
import tensorflow_hub as hub


def main():
    print_setup()
    train_data, valid_data, test_data = download_data()
    explore(train_data)


def print_setup():
    print('TF:', tf.__version__)
    print('Eager:', tf.executing_eagerly())
    print('Hub:', hub.__version__)
    print(
        'GPU:',
        'available' if tf.ocnfig.list_phyisical_devices('GPU')
        else 'not available')


def download_data():
    return ds.load(
        name='imdb_reviews',
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)


def explore(data):
    examples, labels = next(iter(data.batch(10)))
    print(examples)
    print('Labels:')
    print(labels)


if __name__ == '__main__':
    main()
