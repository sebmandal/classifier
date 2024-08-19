from module.helpers import load_data, accuracy
from module.core import Network
import numpy as np


def main():
	inp = input(
	    '1. Load existing model\n2. Train a new model (MAKE A COPY OF EXISTING MODEL)\n> '
	)
	network = Network([784, 128, 10])

	# load the data
	train_X, train_y, test_X, test_y = load_data()

	# flatten the images
	train_X = train_X.reshape(-1, 784)
	test_X = test_X.reshape(-1, 784)

	# normalize the images
	train_X = train_X / 255.0
	test_X = test_X / 255.0

	network.load_data((train_X, train_y, test_X, test_y))

	if inp == '1':
		network.load_model('model.h5')

		inp = input('Do you want to train the model more? (y/n)\n> ')

		if inp == 'y':
			epochs = input('Enter the number of epochs: ')
			learning_rate = input('Enter the learning rate: ')
			network.train(epochs=int(epochs), lr=float(learning_rate))
			network.save_model('model.h5')

	else:
		network.initialize_model()

		# train the network
		network.train(epochs=100, lr=0.01)

		# save the model
		network.save_model('model.h5')

	# make predictions on the test set, print each prediction
	predictions = np.argmax(network.forward(test_X), axis=1)
	print('Predictions:', predictions)
	print('True values:', test_y)

	# calculate the accuracy
	acc = accuracy(test_y, predictions)
	print(f'Accuracy: {acc}')


if __name__ == "__main__":
	main()
