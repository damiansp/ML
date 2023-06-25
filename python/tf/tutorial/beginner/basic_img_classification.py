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
        tf.keras.layers.Flattern(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(LABELS))])
    return mod
                   

if __name__ == '__main__':
    main()
