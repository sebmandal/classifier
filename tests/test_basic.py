import numpy as np
from module.core import Network

network = Network([784, 128, 10])
train_X = np.random.randn(100, 784)
train_y = np.random.randint(0, 10, 100)
test_X = np.random.randn(100, 784)
test_y = np.random.randint(0, 10, 100)
network.load_data((train_X, train_y, test_X, test_y))

# initialize the model
network.initialize_model()
predictions = np.argmax(network.forward(test_X), axis=1)
acc1 = np.mean(test_y == predictions)

# train the model
network.train(epochs=10, lr=0.01)
predictions = np.argmax(network.forward(test_X), axis=1)
acc2 = np.mean(test_y == predictions)

print(acc2 == acc1)
