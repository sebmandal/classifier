import numpy as np
from .helpers import relu, relu_derivative


class Network:

	def __init__(self, nodes):
		"""
		Initialize the network.
		"""
		self.nodes = nodes

	def initialize_model(self):
		"""
		Initialize the model.
		"""
		self.weights = [
		    np.random.randn(self.nodes[i], self.nodes[i + 1])
		    for i in range(len(self.nodes) - 1)
		]
		self.biases = [
		    np.random.randn(self.nodes[i + 1])
		    for i in range(len(self.nodes) - 1)
		]
		print('Model initialized.')

	def save_model(self, filename):
		"""
		Save the model.
		"""
		np.savez(filename, *self.weights, *self.biases)
		print(f'Model saved to {filename}.')

	def load_model(self, filename):
		"""
		Load the model.
		"""
		data = np.load(filename + '.npz')
		num_layers = len(self.nodes) - 1
		self.weights = [data[f'arr_{i}'] for i in range(num_layers)]
		self.biases = [
		    data[f'arr_{i + num_layers}'] for i in range(num_layers)
		]
		print(f'Model loaded from {filename}.')

	def load_data(self, data):
		"""
		Load the data.
		"""
		self.train_X, self.train_y, self.test_X, self.test_y = data
		print('Data loaded.')

	def train(self, epochs=10, lr=0.01):
		"""
		Train the network.
		"""
		for epoch in range(epochs):
			for x, y in zip(self.train_X, self.train_y):
				self.backprop(x, y, lr)
			print(f'Epoch {epoch + 1}/{epochs}')
		print('Training complete.')

	def forward(self, x) -> np.ndarray:
		for w, b in zip(self.weights, self.biases):
			x = relu(np.dot(x, w) + b)
		return x

	def backprop(self, x, y, lr=0.01):
		"""
		Backpropagation.
		"""
		# forward pass
		activations = [x]
		for w, b in zip(self.weights, self.biases):
			x = relu(np.dot(x, w) + b)
			activations.append(x)

		# backward pass
		delta = activations[-1] - y
		for i in range(2, len(self.nodes)):
			delta = np.dot(delta, self.weights[-i + 1].T)
			delta *= relu_derivative(activations[-i])
			delta = np.clip(delta, -1, 1)    # Gradient clipping
			self.weights[-i] -= lr * np.outer(activations[-i - 1], delta)
			self.biases[-i] -= lr * delta

		# update weights and biases for the first layer
		delta = np.clip(delta, -1, 1)    # Gradient clipping
		self.weights[0] -= lr * np.outer(activations[0], delta)
		self.biases[0] -= lr
