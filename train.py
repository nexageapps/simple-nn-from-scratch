import numpy as np
from nn import NeuralNetwork
from xor_dataset import generate_xor_dataset


def evaluate(model, X, y):
    """Evaluate model accuracy."""
    ŷ = model.predict(X)
    accuracy = np.mean(ŷ == y) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    print("\nPredictions:")
    for i, (input_, target, pred) in enumerate(zip(X, y, ŷ)):
        print(f"  Input: {input_} -> Target: {target[0]}, Predicted: {pred[0]}")


def main():
    X, y = generate_xor_dataset()

    print("Training Neural Network on XOR Problem")
    print("=" * 40)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    print()

    net = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    net.train(X, y, epochs=10000, lr=0.8, print_every=1000)

    evaluate(net, X, y)


if __name__ == "__main__":
    main()
