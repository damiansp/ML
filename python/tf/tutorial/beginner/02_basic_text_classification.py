# Derived from the tutorial at
# www.tensorflow.org/tutorials/keras/text_classification
import os
import re
import shutil
import string

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses


print(tf.__version__)


BATCH = 32
EPOCHS = 10
SEED = 42
MAX_FEATURES = 10_000
SEQ_LEN = 250
EMBED_DIM = 16
DROPOUT = 0.2


def main():
    raw_dataset = download_data()
    [train_ds, val_ds, test_ds], vectorize_layer = prep_data(raw_dataset)
    mod = create_model()
    history = train(mod, train_ds, val_ds)
    evaluate(mod, test_ds)
    plot_history(history.history)
    export_model(mod, vectorize_layer, raw_dataset[-1])

    
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


def prep_data(raw_dataset):
    raw_train_ds, raw_val_ds, raw_test_ds = raw_dataset
    vectorize_layer = layers.TextVectorization(
        strandardize=standardize_text,
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=SEQ_LEN)
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    print('Review:', first_review)
    print('Label:', raw_train_ds.class_names[first_label])
    print('Vectorized:', vectorize_text(first_review, first_label))
    print('1287 --> ', vectorize_layer.get_vocabulary()[1287])
    print(' 313 --> ', vectorize_layer.get_vocabulary()[313])
    print('Vocab size:', len(vectorize_layer.get_vocabulary()))
    return (
        [ds.map(vectorize_text)
         for ds in [raw_train_ds, raw_val_ds, raw_test_ds]].
        vectorize_layer)


def standardize_text(data):
    lower = tf.strings.lower(data)
    no_html = tf.strings.regex_replace(lower, '<br />', ' ')
    return tf.strings.regex_replace(
        no_html, '[%s]' % re.escape(string.punctuation), '')


def create_model():
    mod = tf.keras.Sequential([
        layers.Embedding(MAX_FEATURES + 1, EMBED_DIM),
        layers.Dropout(DROPOUT),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(DROPOUT),
        layers.Dense(1)])
    print(mod.summary())
    mod.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=tf.metrics.BinaryAccuracy(threshold=0.))
    return mod


def train(mod, train_ds, val_ds):
    history = mod.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    return history


def evaluate(mod, test_ds):
    loss, acc = mod.evaluate(test_ds)
    print('Loss:', loss)
    print('Acc: ', acc)


def plot_history(history):
    acc = history['binary_accuracy']
    val_acc = history['val_binary_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    plt.subplot(1, 2, 1)
    plt.plot(loss, color='k', label='training')
    plt.plot(val_loss, color='r', label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(acc, color='k')
    plt.plot(val_acc, color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def export_model(mod, vectorize_layer, raw_test_set):
    mod = tf.keras.Sequential([
        vectorize_layer, mod, layers.Activation('sigmoid')])
    mod.compile(
        loss=losses.BinaryCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy'])
    loss, acc = mod.evaluate(raw_test_set)
    print('Test acc:', acc)
    new_exs = [
        'The movie was great!', 'The movie was ok',
        'That was the worst movie ever!']
    print('Preds on new data:', mod.predict(new_exs))
    
