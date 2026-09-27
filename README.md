# Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

[![CI](https://github.com/sanjayk36725/sutton-and-barto-from-scratch-1-bandits-and-dynamic-programming/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjayk36725/sutton-and-barto-from-scratch-1-bandits-and-dynamic-programming/actions/workflows/ci.yml)
[![CodeQL](https://github.com/sanjayk36725/sutton-and-barto-from-scratch-1-bandits-and-dynamic-programming/actions/workflows/codeql.yml/badge.svg)](https://github.com/sanjayk36725/sutton-and-barto-from-scratch-1-bandits-and-dynamic-programming/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implement core multi-armed bandit algorithms and dynamic-programming methods from Sutton and Barto. Build stationary and nonstationary bandit testbeds, compare epsilon-greedy, optimistic, UCB and gradient strategies, then solve gridworld and gambler MDPs with policy and value iteration.

## Tech stack

- Python 3.10+
- NumPy
- pytest + pytest-cov
- GitHub Actions
- CodeQL

## How to run

```bash
python -m pip install -r requirements.txt
python scaffold.py
python -m pytest -q
```

## Implemented algorithms

### Multi-armed bandits
- [x] create_bandit_testbed
- [x] pull_arm
- [x] sample_average_update
- [x] epsilon_greedy_action
- [x] run_bandit_episode
- [x] track_rewards_and_optimal_actions
- [x] average_bandit_curves
- [x] apply_random_walk_drift
- [x] constant_step_size_update
- [x] optimistic_initialization
- [x] ucb_action_select
- [x] gradient_bandit_update
- [x] bandit_parameter_study

### Dynamic programming
- [x] build_gridworld_mdp
- [x] iterative_policy_evaluation
- [x] greedy_policy_improvement
- [x] policy_iteration
- [x] value_iteration
- [x] build_gambler_mdp
- [x] gambler_value_iteration
- [x] extract_optimal_stakes

## Project structure

```text
.
├── model.py
├── scaffold.py
├── tests/
│   ├── test_model.py
│   └── test_validation.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── docs/
└── .github/workflows/
    ├── ci.yml
    └── codeql.yml
```

## Quality checks

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12, checks Python compilation, enforces 90% minimum coverage, runs the end-to-end demonstration, and stores the coverage report. CodeQL performs automated Python security analysis for pushes and pull requests targeting `main`.

The implementation uses explicit random seeds for reproducible experiments and bounds public simulation sizes before NumPy allocations.

## Reference

Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd edition.

Built as a learning implementation, with reinforcement-learning equations translated into executable Python.
