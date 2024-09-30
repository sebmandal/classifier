import unittest
import numpy as np
from module import (
    Network,
    relu,
    relu_derivative,
    softmax,
    compute_loss,
    preprocess_data,
)
from sklearn.datasets import fetch_openml


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.input_size = 4
        self.hidden_size = 3
        self.output_size = 2
        self.learning_rate = 0.1
        self.mlp = Network(
            self.input_size, self.hidden_size, self.output_size, self.learning_rate
        )

        self.X_test = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        self.y_test = np.array([[1, 0], [0, 1]])

    def test_forward_propagation(self):
        output = self.mlp.forward(self.X_test)
        self.assertEqual(output.shape, (2, self.output_size))

    def test_backward_propagation(self):
        original_W1 = np.copy(self.mlp.W1)
        original_W2 = np.copy(self.mlp.W2)

        self.mlp.forward(self.X_test)
        self.mlp.backward(self.X_test, self.y_test)

        self.assertFalse(np.array_equal(original_W1, self.mlp.W1))
        self.assertFalse(np.array_equal(original_W2, self.mlp.W2))


class TestActivations(unittest.TestCase):

    def test_relu(self):
        Z = np.array([[1, -1], [0, 5]])
        A = relu(Z)
        expected = np.array([[1, 0], [0, 5]])
        np.testing.assert_array_equal(A, expected)

    def test_relu_derivative(self):
        Z = np.array([[1, -1], [0, 5]])
        dA = relu_derivative(Z)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(dA, expected)

    def test_softmax(self):
        Z = np.array([[1, 2], [3, 4]])
        A = softmax(Z)
        expected = np.array(
            [
                [
                    np.exp(1) / (np.exp(1) + np.exp(2)),
                    np.exp(2) / (np.exp(1) + np.exp(2)),
                ],
                [
                    np.exp(3) / (np.exp(3) + np.exp(4)),
                    np.exp(4) / (np.exp(3) + np.exp(4)),
                ],
            ]
        )
        np.testing.assert_almost_equal(A, expected, decimal=5)


class TestLoss(unittest.TestCase):

    def test_loss_computation(self):
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.7, 0.3], [0.4, 0.6]])
        loss = compute_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, -np.log(0.7) / 2 - np.log(0.6) / 2, places=5)


class TestUtils(unittest.TestCase):

    def test_preprocess_data(self):
        mnist = fetch_openml("mnist_784")
        X_train, X_test, y_train, y_test = preprocess_data(mnist)

        self.assertEqual(X_train.shape[1], 784)
        self.assertEqual(y_train.shape[1], 10)


if __name__ == "__main__":
    unittest.main()
