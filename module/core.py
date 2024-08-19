import numpy as np
import json


class Network:

	def __init__(self, nodes):
		"""
		Initialize the network.
		"""
		self.nodes = nodes

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

	def load_data(self, data):
		"""
		Load the data.
		"""
		self.train_X, self.train_y, self.test_X, self.test_y = data
		print('Data loaded.')

	def forward(self, x):
		"""
		Forward pass.
		"""
		for w, b in zip(self.weights, self.biases):
			x = np.dot(x, w) + b
		return x

	def train(self, epochs=10, lr=0.01):
		"""
		Train the network.
		"""
		for epoch in range(epochs):
			for x, y in zip(self.train_X, self.train_y):
				self.backprop(x, y, lr)
			print(f'Epoch {epoch + 1}/{epochs}')
		print('Training complete.')

	def backprop(self, x, y, lr=0.01):
		"""
		Backpropagation.
		"""
		# forward pass
		activations = [x]
		for w, b in zip(self.weights, self.biases):
			x = np.dot(x, w) + b
			activations.append(x)

		# backward pass
		delta = activations[-1] - y
		for i in range(2, len(self.nodes)):
			delta = np.dot(delta, self.weights[-i + 1].T)
			activations[-i] -= lr * delta
		# print('Backpropagation complete.')
