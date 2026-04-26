import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid activation function."""
    s = sigmoid(x)
    return s * (1 - s)


class NeuralNetwork:
    """
    A simple neural network with one hidden layer.

    Architecture:
        Input Layer: 2 neurons
        Hidden Layer: 4 neurons (sigmoid activation)
        Output Layer: 1 neuron (sigmoid activation)
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with random weights.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
        """
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input data of shape (batch_size, input_size)

        Returns:
            Output predictions of shape (batch_size, output_size)
        """
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, lr):
        """
        Backward pass to compute gradients and update weights.

        Args:
            X: Input data
            y: True labels
            lr: Learning rate
        """
        m = X.shape[0]

        dL_dz2 = (self.a2 - y) * sigmoid_derivative(self.z2)
        dW2 = self.a1.T @ dL_dz2 / m
        db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m

        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * sigmoid_derivative(self.z1)

        dW1 = X.T @ dL_dz1 / m
        db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs, lr, print_every=500):
        """
        Train the neural network.

        Args:
            X: Training data
            y: True labels
            epochs: Number of training iterations
            lr: Learning rate
            print_every: Print loss every n epochs
        """
        for epoch in range(epochs):
            ŷ = self.forward(X)
            loss = np.mean((y - ŷ) ** 2)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

            self.backward(X, y, lr)

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input data

        Returns:
            Binary predictions (0 or 1)
        """
        ŷ = self.forward(X)
        return (ŷ >= 0.5).astype(int)
