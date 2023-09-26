import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


np.set_printoptions(precision=3, suppress=True)
print('TF:', tf.__version__)


def main():
    raw_data = get_data()
    data = clean(raw_data)
    train_ds, test_ds = split_train_test(data)
    inspect(train_ds)
    X_train, y_train = split_xy(train_ds, 'MPG')
    X_test, y_test = split_xy(train_ds, 'MPG')
    normalizer = init_normalizer(X_train)
    

def get_data():
    url = (
        'http://archive.ics.uci.edu/ml/machine-learning-database/auto-mpg.data'
    )
    col_names = [
        'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
        'Acceleration', 'Model Year', 'Origin']
    return pd.read_csv(
        url,
        names=col_names,
        na_values='?',
        comment='\t',
        sep=' ',
        skipinitialspace=True)


def clean(df):
    print(df.isna().sum())
    df.dropna(inplace=True)
    df = dummify_origin(df)
    return df


def dummify_origin(df):
    df.Origin = df.Origin.map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    df = pd.get_dummies(
        df, columns=['Origin'], prefix='Origin', prefix_sep='_')
    return df


def split_train_test(data):
    train = data.sample(frac=0.8, random_state=5093)
    test = data.drop(train.index)
    return train, test


def inspect(train_ds):
    sns.pairplot(
        train_ds[['MPG', 'Cylinders', 'Displacement', 'Weight']],
        diag_kind='kde')
    print(train_ds.describe().transpose())
    

def split_xy(ds, y_name):
    X = ds.copy()
    y = X.pop(y_name)
    return X, y


def init_normalizer(X_train):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))
    return normalizer

    
if __name__ == '__main__':
    main()

