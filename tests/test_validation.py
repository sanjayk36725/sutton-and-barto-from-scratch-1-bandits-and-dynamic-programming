import numpy as np
import pytest

from model import (
    MAX_BANDIT_ARMS,
    MAX_GAMBLER_GOAL,
    MAX_GRID_CELLS,
    MAX_SIMULATION_STEPS,
    apply_random_walk_drift,
    bandit_parameter_study,
    build_gambler_mdp,
    build_gridworld_mdp,
    constant_step_size_update,
    create_bandit_testbed,
    epsilon_greedy_action,
    extract_optimal_stakes,
    gradient_bandit_update,
    optimistic_initialization,
    run_bandit_episode,
    track_rewards_and_optimal_actions,
)


def test_size_validation_rejects_unsafe_or_invalid_values():
    with pytest.raises(ValueError):
        create_bandit_testbed(0, 0)
    with pytest.raises(ValueError):
        create_bandit_testbed(MAX_BANDIT_ARMS + 1, 0)
    with pytest.raises(TypeError):
        create_bandit_testbed(1.5, 0)
    with pytest.raises(ValueError):
        run_bandit_episode(np.array([0.0]), MAX_SIMULATION_STEPS + 1, 0.1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        build_gridworld_mdp(101, 101)
    with pytest.raises(ValueError):
        build_gambler_mdp(MAX_GAMBLER_GOAL + 1)


def test_bandit_validation_and_empty_step_behavior():
    rng = np.random.default_rng(0)
    assert run_bandit_episode(np.array([0.0]), 0, 0.0, rng).size == 0
    rewards, optimal = track_rewards_and_optimal_actions(np.array([0.0]), 0, 0.0, rng)
    assert rewards.size == 0
    assert optimal.size == 0

    with pytest.raises(ValueError):
        run_bandit_episode(np.array([0.0]), -1, 0.0, rng)
    with pytest.raises(ValueError):
        epsilon_greedy_action(np.array([]), 0.0, rng)
    with pytest.raises(ValueError):
        epsilon_greedy_action(np.array([0.0]), -0.1, rng)
    with pytest.raises(ValueError):
        epsilon_greedy_action(np.array([0.0]), 1.1, rng)


def test_drift_and_step_update_validation():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 2.0])
    assert np.array_equal(apply_random_walk_drift(values, 0.0, rng), values)
    with pytest.raises(ValueError):
        apply_random_walk_drift(values, -0.1, rng)
    with pytest.raises(ValueError):
        constant_step_size_update(values, 0, 1.0, 0.0)
    with pytest.raises(ValueError):
        constant_step_size_update(values, 0, 1.0, 1.1)
    assert np.all(optimistic_initialization(2, 3.0) == 3.0)


def test_gradient_update_changes_preferences():
    updated = gradient_bandit_update(np.zeros(3), 1, 2.0, 1.0, 0.1)
    assert updated[1] > updated[0]
    assert not np.allclose(updated, 0.0)


def test_parameter_study_covers_all_methods():
    settings = [
        {"method": "epsilon_greedy", "param": 0.1},
        {"method": "optimistic", "param": 1.0},
        {"method": "ucb", "param": 2.0},
        {"method": "gradient", "param": 0.1},
    ]
    results = bandit_parameter_study(1, 5, 0, settings)
    assert [row["method"] for row in results] == [s["method"] for s in settings]
    assert all(np.isfinite(row["average_reward"]) for row in results)

    with pytest.raises(ValueError):
        bandit_parameter_study(1, 2, 0, [{"method": "unknown", "param": 1.0}])


def test_gambler_input_validation_and_policy_shape():
    with pytest.raises(ValueError):
        build_gambler_mdp(1, 0.4)
    with pytest.raises(ValueError):
        build_gambler_mdp(10, -0.1)
    with pytest.raises(ValueError):
        build_gambler_mdp(10, 1.1)
    with pytest.raises(ValueError):
        extract_optimal_stakes(np.zeros(10), goal=10, head_prob=0.4)
