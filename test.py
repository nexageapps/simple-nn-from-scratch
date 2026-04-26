import numpy as np
from train import train
from xor_dataset import generate_xor_dataset


EXPECTED = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}


def evaluate(model, X, y):
    ŷ = model.predict(X)
    accuracy = np.mean(ŷ == y) * 100

    print("\nPredictions")
    print("-" * 44)
    print(f"  {'x1':>4} {'x2':>4} {'Target':>8} {'Predicted':>10} {'':>6}")
    print("-" * 44)
    for input_, target, pred in zip(X, y, ŷ):
        status = "PASS" if target[0] == pred[0] else "FAIL"
        print(f"  {input_[0]:>4} {input_[1]:>4} {target[0]:>8} {pred[0]:>10}   {status}")
    print("-" * 44)
    print(f"  Accuracy: {accuracy:.2f}%")
    return accuracy


def validate(model, X, y):
    print("\nValidation")
    print("-" * 44)
    passed, failed = 0, 0
    for input_, target in zip(X, y):
        pred = model.predict(input_.reshape(1, -1))[0][0]
        key = (int(input_[0]), int(input_[1]))
        expected = EXPECTED[key]
        ok = pred == expected
        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok
        print(f"  XOR{key} -> expected {expected}, got {pred}  [{status}]")
    print("-" * 44)
    print(f"  {passed} passed, {failed} failed")
    assert failed == 0, f"Validation failed: {failed} test(s) did not match expected output"
    print("  All validations passed.")


if __name__ == "__main__":
    net = train()
    X, y = generate_xor_dataset()
    evaluate(net, X, y)
    validate(net, X, y)
