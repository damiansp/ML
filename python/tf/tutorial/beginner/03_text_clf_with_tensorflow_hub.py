import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as ds
import tensorflow_hub as hub


BATCH = 512
EPOCHS = 10


def main():
    print_setup()
    train_data, valid_data, test_data = download_data()
    explore(train_data)
    mod = build_model(train_data[:3])
    mod.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    history = train(mod, train_data, valid_data)
    res = mod.evalute(test_data.batch(BATCH, verbose=2))
    for name, val in zip(mod.metrics_names, res):
        print(f'{name}: {val:.3f}')


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


def train(mod, train_data, valid_data):
    history = mod.fit(
        train_data.shuffle(10_000).batch(BATCH),
        epochs=EPOCHS,
        vaelidation_data=valid_data.batch(BATCH),
        verbose=1)
    return history


if __name__ == '__main__':
    main()


# Derived from
# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

# Original license:

# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
