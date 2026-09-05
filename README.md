# Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

[![CI](https://github.com/sanjayk36725/sutton-and-barto-from-scratch-1-bandits-and-dynamic-programming/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjayk36725/sutton-and-barto-from-scratch-1-bandits-and-dynamic-programming/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch Python implementation of the first reinforcement-learning foundations in Sutton & Barto: k-armed bandits, exploration strategies, and dynamic programming for small Markov decision processes.

The code intentionally uses NumPy and the Python standard library rather than a reinforcement-learning framework, so the update equations and control loops remain visible.

## Implemented topics

### Multi-armed bandits
- Stationary k-armed testbed and noisy arm pulls
- Sample-average action-value estimation
- Epsilon-greedy action selection
- Reward and optimal-action tracking
- Multi-run learning curves
- Random-walk nonstationarity
- Constant-step-size updates
- Optimistic initial values
- Upper-confidence-bound (UCB) action selection
- Gradient bandit preference updates
- Parameter-study runner

### Dynamic programming
- Deterministic gridworld MDP construction
- Iterative policy evaluation
- Greedy policy improvement
- Policy iteration
- Value iteration
- Gambler's problem MDP and value iteration
- Optimal-stake extraction

The implementations follow the concepts in *Reinforcement Learning: An Introduction*, 2nd edition, by Richard S. Sutton and Andrew G. Barto.

## Quick start

```bash
python -m pip install -r requirements.txt
python scaffold.py
python -m pytest -q
```

## Project layout

```text
.
├── model.py                         # Algorithms and MDP helpers
├── scaffold.py                      # End-to-end demonstration
├── tests/test_model.py              # Core regression tests
├── tests/test_validation.py         # Validation and edge-case tests
├── requirements.txt                 # Runtime/test dependencies
├── pyproject.toml                   # Project metadata and pytest config
├── LICENSE                           # MIT license
├── docs/                             # Browser-friendly project documentation
└── .github/workflows/ci.yml         # Automated test workflow
```

## Reproducibility and safety

Experiments accept explicit seeds. Use a fixed seed when comparing algorithm changes so that implementation changes are easier to isolate from random variation.

Public size parameters are bounded before NumPy allocations are created. This keeps educational simulation code from accepting unbounded allocation requests.

## Development

Run the test suite and coverage locally before submitting a change:

```bash
python -m pytest --cov=model --cov-report=term-missing -q
```

GitHub Actions runs the test suite across Python 3.10, 3.11, and 3.12.

## Reference

Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd edition. The authors' complete draft is available from the official book site.

Built as a learning implementation, with equations translated directly into executable Python.
