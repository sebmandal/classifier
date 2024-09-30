from sklearn.datasets import fetch_openml
from module.core import Network
from module.trainer import Trainer
from module.utils import preprocess_data

# Load data
mnist = fetch_openml("mnist_784")
X_train, X_test, y_train, y_test = preprocess_data(mnist)

# Initialize model
input_size = 784
hidden_size = 64
output_size = 10
learning_rate = 0.1

network = Network(input_size, hidden_size, output_size, learning_rate)
trainer = Trainer(network)

# Train the model
trainer.train(X_train, y_train, epochs=100)

# Evaluate the model
accuracy = trainer.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
