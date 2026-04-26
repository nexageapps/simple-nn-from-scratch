import numpy as np


def generate_xor_dataset():
    """
    Generate XOR dataset.

    XOR is a classic binary classification problem where the output is 1
    if the inputs are different, and 0 if they are the same.

    Returns:
        X: Input features of shape (4, 2)
        y: Output labels of shape (4, 1)
    """
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    return X, y
