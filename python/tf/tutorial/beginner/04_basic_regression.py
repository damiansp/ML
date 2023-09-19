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
    # t/t split
    

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
    


if __name__ == '__main__':
    main()

