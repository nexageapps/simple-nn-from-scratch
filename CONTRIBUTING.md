# Contributing

Thanks for wanting to help. Please send changes as a pull request; do not push directly to `main`.

1. Fork the repository.
2. Create a branch from `main`.
3. Make your change.
4. Open a pull request against `main`.

A maintainer has to approve a pull request before it can be merged.

## Checks

Pull request CI installs `requirements.txt` and runs `python test.py`. That script trains the small XOR network (10,000 epochs) and then asserts the four truth-table outputs; a failed assertion exits non-zero.

Workflows on a pull request from a first-time fork need a maintainer to approve them before they will run.
