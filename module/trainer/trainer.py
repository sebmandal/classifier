import numpy as np
from module.loss import compute_loss


class Trainer:
    def __init__(self, model, learning_rate=0.1):
        self.model = model

    def train(self, X_train, y_train, epochs=100):
        for epoch in range(epochs):
            y_pred = self.model.forward(X_train)
            loss = compute_loss(y_train, y_pred)
            self.model.backward(X_train, y_train)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.forward(X_test)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
        return accuracy

    def predict(self, X):
        return np.argmax(self.model.forward(X), axis=1)
