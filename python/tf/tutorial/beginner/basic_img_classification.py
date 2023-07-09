import ssl

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)
ssl._create_default_https_context = ssl._create_unverified_context


DATA = '/Users/damiansp/Learning/ML/data'
LABELS = [
    'T-shirt/Top', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot']
EPOCHS = 10


def main():
    (X_train, y_train), (X_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data())
    explore(X_train, y_train, X_test, y_test)
    X_train, X_test = preprocess(X_train, X_test, y_train)
    mod = init_mod()
    mod.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    mod.fit(X_train, y_train, epochs=EPOCHS)
    test_loss, test_acc = mod.evaluate(X_test, y_test, verbose=2)
    print(f'Test acc: {test_acc:.4f}')
    prob_mod = tf.keras.Sequential([mod, tf.keras.layers.Softmax()])
    preds = prob_mod.predict(X_test)
    print('Sample preds:', preds[0])
    print('Argmax:', np.argmax(preds[0]))
    print('Actual:', y_test[0])
    verify_predictions(preds, X_test, y_test)
    use_mod(X_test, y_test, mod)


def explore(X_train, y_train, X_test, y_test):
    print('X_txrain:', X_train.shape)
    print('y_train:', len(y_train), y_train)
    print('X_test:', X_test.shape)
    print('y_test:', len(y_test))
    
    
def preprocess(X_train, X_test, y_train):
    plot_img(X_train[0])
    X_train = X_train / 255.
    X_test = X_test / 255.
    plot_sample(X_train, y_train)
    return X_train, X_test


def plot_img(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def plot_sample(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i], cmap=plt.cm.binary)
        plt.xlabel(LABELS[y[i]])
    plt.show()


def init_mod():
    mod = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(LABELS))])
    return mod


def plot_image(i, pred_array, y_actual, img):
    y_actual, img = y_actual[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    y_pred = np.argmax(pred_array)
    color = 'blue' if y_pred == y_actual else 'red'
    plt.xlabel(
        f'{LABELS[y_pred]} {100 * np.max(pred_array):2.0f}% '
        f'({LABELS[y_pred]})',
        color=color)


def plot_val_array(i, pred_array, y_actual):
    y_actual = y_actual[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), pred_array, color='#777777')
    plt.ylim([0, 1])
    y_pred = np.argmax(pred_array)
    this_plot[y_pred].set_color('red')
    this_plot[y_actual].set_color('blue')


def verify_predictions(preds, X_test, y_test):
    n_rows = 5
    n_cols = 3
    n_imgs = n_rows * n_cols
    plt.figure(figsize=[2 * 2 * n_cols, 2 * n_rows])
    for i in range(n_imgs):
        plt.subplot(n_rows, 2 * n_cols, 2*i + 1)
        plot_image(i, preds[i], y_test, X_test)
        plt.subplot(n_rows, 2 * n_cols, 2*i + 2)
        plot_val_array(i, preds[i], y_test)
    plt.show()


def use_mod(X_test, y_test, mod):
    img = X_test[1]
    print(img.shape)  # 28, 28
    # Add img to batch of one:
    img = np.expand_dims(img, 0)
    print(img.shape)  # 1, 28, 28
    pred = mod.predict(img)
    print(pred)       # probs for each class
    plot_val_array(1, pred[0], y_test)
    _ = plt.xticks(range(10), LABELS, rotation=45)
    plt.show()
    

if __name__ == '__main__':
    main()


# Derived from:
# https://www.tensorflow.org/tutorials/keras/classification#use_the_trained_model
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
