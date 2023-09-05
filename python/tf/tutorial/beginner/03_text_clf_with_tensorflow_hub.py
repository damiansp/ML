import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as ds
import tensorflow_hub as hub


def main():
    print_setup()
    train_data, valid_data, test_data = download_data()
    explore(train_data)
    mod = build_model(train_data[:3])
    mod.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    ## train()


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


def build_model(training_sample):
    embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'
    hub_layer = hub.KerasLayer(
        embedding, input_shape=[], dtype=tf.string, trainable=True)
    print(hub_layer(training_sample))
    mod = tf.keras.Sequential()
    mod.add(hub_layer)
    mod.add(tf.keras.layers.Dense(16, activation='relu'))
    mod.add(tf.keras.layers.Dense(1))
    print(mod.summary())
    return mod


if __name__ == '__main__':
    main()
