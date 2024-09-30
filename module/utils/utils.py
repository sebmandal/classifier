from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np


def one_hot_encode(y):
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(y)


def preprocess_data(mnist_data):
    X = mnist_data.data / 255.0  # Normalize
    y = mnist_data.target.astype(int).values.reshape(-1, 1)
    y_encoded = one_hot_encode(y)
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42)
