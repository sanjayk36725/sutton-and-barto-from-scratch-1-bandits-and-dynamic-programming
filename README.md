# Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

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
- Small parameter-study runner

### Dynamic programming
- Deterministic gridworld MDP construction
- Iterative policy evaluation
- Greedy policy improvement
- Policy iteration
- Value iteration
- Gambler's problem MDP and value iteration
- Optimal-stake extraction

The implementations follow the concepts in *Reinforcement Learning: An Introduction*, 2nd edition, by Richard S. Sutton and Andrew G. Barto. citeturn0search0

## Quick start

```bash
python -m pip install -r requirements.txt
python scaffold.py
python -m pytest -q
```

## Project layout

```text
.
├── model.py                 # Algorithms and MDP helpers
├── scaffold.py              # End-to-end demonstration
├── tests/test_model.py      # Regression and behavior tests
├── requirements.txt         # Runtime/test dependencies
├── pyproject.toml           # Project metadata and pytest config
├── docs/                    # Browser-friendly project documentation
└── .github/workflows/ci.yml # Automated test workflow
```

## Reproducibility

Experiments accept explicit seeds. Use a fixed seed when comparing algorithm changes so that changes in the implementation are easier to isolate from random variation.

## Development

Run the test suite before submitting a change:

```bash
python -m pytest -q
```

The CI workflow runs the tests and the end-to-end scaffold on Python 3.10, 3.11, and 3.12.

## Reference

Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd edition. The authors' complete draft is available from the official book site. urlSutton & Barto book drafthttps://www.incompleteideas.net/book/bookdraft2018mar21.pdf

Built as a learning implementation, with equations translated directly into executable Python.