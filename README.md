# Simple Neural Network from Scratch

A minimal implementation of a neural network with one hidden layer, designed to solve the XOR problem using only NumPy.

## Architecture

```mermaid
flowchart LR
    subgraph IL["Input Layer"]
        x1(["x1"])
        x2(["x2"])
    end

    subgraph HL["Hidden Layer  |  Sigmoid"]
        h1(["h1"])
        h2(["h2"])
        h3(["h3"])
        h4(["h4"])
    end

    subgraph OL["Output Layer  |  Sigmoid"]
        y(["y"])
    end

    x1 -- "W1 + b1" --> h1
    x1 -- "W1 + b1" --> h2
    x1 -- "W1 + b1" --> h3
    x1 -- "W1 + b1" --> h4
    x2 -- "W1 + b1" --> h1
    x2 -- "W1 + b1" --> h2
    x2 -- "W1 + b1" --> h3
    x2 -- "W1 + b1" --> h4

    h1 -- "W2 + b2" --> y
    h2 -- "W2 + b2" --> y
    h3 -- "W2 + b2" --> y
    h4 -- "W2 + b2" --> y

    style x1 fill:#4A90D9,stroke:#2c5f8a,color:#fff
    style x2 fill:#4A90D9,stroke:#2c5f8a,color:#fff
    style h1 fill:#E8A838,stroke:#a87020,color:#fff
    style h2 fill:#E8A838,stroke:#a87020,color:#fff
    style h3 fill:#E8A838,stroke:#a87020,color:#fff
    style h4 fill:#E8A838,stroke:#a87020,color:#fff
    style y  fill:#5BAD6F,stroke:#357a47,color:#fff
```

| Layer | Neurons | Weights | Bias | Activation |
|-------|---------|---------|------|------------|
| Input | 2 | - | - | None |
| Hidden | 4 | W1 (2x4) | b1 | Sigmoid |
| Output | 1 | W2 (4x1) | b2 | Sigmoid |

- Loss Function: Mean Squared Error (MSE)
- Optimization: Gradient Descent with backpropagation

## XOR Problem

The XOR function is not linearly separable, requiring at least one hidden layer to solve.

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

## Files

- `nn.py` - Neural network implementation
- `xor_dataset.py` - XOR dataset generation
- `train.py` - Training script

## Usage

```bash
python3 train.py
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10,000 |
| Learning rate | 0.8 |
| Hidden neurons | 4 |
| Seed | 42 |

## Mathematical Background

### Forward Pass

```
z1 = X . W1 + b1       a1 = sigmoid(z1)
z2 = a1 . W2 + b2      a2 = sigmoid(z2)
```

### Backward Pass

```
dL/dz2 = (a2 - y) * sigmoid'(z2)
dW2    = a1^T . dL/dz2 / m
dL/dz1 = (dL/dz2 . W2^T) * sigmoid'(z1)
dW1    = X^T . dL/dz1 / m
```

## Requirements

- Python 3.x
- NumPy

## License

MIT License
