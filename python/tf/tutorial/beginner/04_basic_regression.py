import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


np.set_printoptions(precision=3, suppress=True)
print('TF:', tf.__version__)


ETA = 0.1
EPOCHS = 100


def main():
    raw_data = get_data()
    data = clean(raw_data)
    train_ds, test_ds = split_train_test(data)
    inspect(train_ds)
    X_train, y_train = split_xy(train_ds, 'MPG')
    X_test, y_test = split_xy(train_ds, 'MPG')
    normalizer = init_normalizer(X_train)
    hp_mod = run_one_var_regression(X_train, y_train)
    test_res = {}
    test_res = evaluate_mod(
        hp_mod, 'hp_mod', X_test['Horsepower'], y_test, test_res)
    plot_hp_mod(hp_mod, X_train, y_train)
    

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


def run_one_var_regression(X_train, y_train):
    hp = np.array(X_train['Horsepower'])
    hp_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    hp_normalizer.adapat(hp)
    hp_mod = tf.keras.Sequential([hp_normalizer, layers.Dense(units=1)])
    print(hp_mod.summary())
    hp_mod.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ETA),
        loss='mean_absolute_error')
    history = hp_mod.fit(
        X_train['Horsepower'],
        y_train,
        epochs=EPOCHS,
        verbose=0,
        validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    plot_loss(history)
    return hp_mod


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    #plt.ylim([0, 10)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.show()


def evaluate_mod(mod, mod_name, X_test, y_test, test_res):
    test_res[mod_name] = mod.evaluate(X_test, y_test, verbose=0)
    return test_res


def plot_hp_mod(hp_mod, X_train, y_train):
    x = tf.linspace(0., 250, 251)
    fitted = hp_mod.predict(x)
    plt.scatter(X_train['Horsepower'], y_train, label='Data')
    plt.plot(x, fitted, color='k', label='Fitted model')
    plt.xlabel('HP')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

