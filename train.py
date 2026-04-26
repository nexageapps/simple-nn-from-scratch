import numpy as np
from nn import NeuralNetwork
from xor_dataset import generate_xor_dataset


def train():
    X, y = generate_xor_dataset()

    print("Training Neural Network on XOR Problem")
    print("=" * 40)
    print(f"  Input shape : {X.shape}")
    print(f"  Output shape: {y.shape}")
    print()

    net = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    net.train(X, y, epochs=10000, lr=0.8, print_every=1000)

    print("\nTraining complete.")
    return net


if __name__ == "__main__":
    train()
