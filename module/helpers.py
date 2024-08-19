from keras.datasets import mnist
import numpy as np


def load_data() -> tuple:
	"""
	Load and return the dataset.
	"""
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	print('x_train shape:', x_train.shape)    # (60000, 28, 28)
	return x_train, y_train, x_test, y_test


def accuracy(y_true, y_pred) -> float:
	"""
	Calculate the accuracy of the predictions.
	"""
	return np.mean(y_true == y_pred)


def relu(x):
	return np.maximum(0, x)


def relu_derivative(x):
	return np.where(x > 0, 1, 0)
