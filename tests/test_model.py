import numpy as np

from model import (
    average_bandit_curves,
    build_gambler_mdp,
    build_gridworld_mdp,
    constant_step_size_update,
    create_bandit_testbed,
    epsilon_greedy_action,
    extract_optimal_stakes,
    gambler_value_iteration,
    greedy_policy_improvement,
    optimistic_initialization,
    policy_iteration,
    sample_average_update,
    ucb_action_select,
    value_iteration,
)


def test_bandit_testbed_is_reproducible():
    assert np.array_equal(create_bandit_testbed(5, 7), create_bandit_testbed(5, 7))


def test_sample_average_update():
    q, counts = sample_average_update(np.zeros(2), np.zeros(2, dtype=int), 1, 4.0)
    q, counts = sample_average_update(q, counts, 1, 2.0)
    assert counts.tolist() == [0, 2]
    assert q[1] == 3.0


def test_epsilon_zero_is_greedy():
    rng = np.random.default_rng(0)
    assert epsilon_greedy_action(np.array([1.0, 3.0, 2.0]), 0.0, rng) == 1


def test_average_curve_shapes():
    rewards, optimal = average_bandit_curves(5, 3, 20, 0.1, seed=1)
    assert rewards.shape == (20,)
    assert optimal.shape == (20,)
    assert np.all((optimal >= 0) & (optimal <= 1))


def test_constant_step_and_optimistic_init():
    q = constant_step_size_update(np.zeros(2), 0, 10.0, 0.5)
    assert q.tolist() == [5.0, 0.0]
    assert np.all(optimistic_initialization(3, 2.0) == 2.0)


def test_ucb_selects_unseen_action():
    assert ucb_action_select(np.array([5.0, 1.0]), np.array([2, 0]), 2, 2.0) == 1


def test_gridworld_algorithms_agree():
    mdp = build_gridworld_mdp()
    vi_values, vi_policy = value_iteration(mdp, gamma=0.9, theta=1e-6)
    pi_values, pi_policy = policy_iteration(mdp, gamma=0.9, theta=1e-6)
    assert np.allclose(vi_values, pi_values, atol=1e-5)
    assert np.array_equal(vi_policy, pi_policy)
    improved = greedy_policy_improvement(mdp, vi_values, gamma=0.9)
    assert np.array_equal(improved, vi_policy)


def test_gambler_value_and_policy_bounds():
    mdp = build_gambler_mdp(20, 0.4)
    values = gambler_value_iteration(20, 0.4, theta=1e-8)
    stakes = extract_optimal_stakes(values, 20, 0.4)
    assert len(values) == 21
    assert values[0] == 0.0
    assert values[20] == 1.0
    for state, actions in mdp["actions"].items():
        assert 0 <= stakes[state] <= max(actions)
