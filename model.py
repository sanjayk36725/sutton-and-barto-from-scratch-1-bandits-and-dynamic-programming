"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    rng = np.random.RandomState(seed)
    return rng.normal(loc=mean, scale=std, size=k)

# Step 2 - pull_arm
def pull_arm(true_values, action, rng):
    return true_values[action] + rng.normal()

# Step 3 - sample_average_update
def sample_average_update(q_values, action_counts, action, reward):
    q_new = q_values.copy()
    c_new = action_counts.copy()

    c_new[action] += 1
    q_new[action] += (reward - q_new[action]) / c_new[action]

    return q_new, c_new

# Step 4 - epsilon_greedy_action
def epsilon_greedy_action(q_values, epsilon, rng):
    k = len(q_values)

    if rng.random() < epsilon:
        return int(rng.integers(0, k))

    return int(np.argmax(q_values))

# Step 5 - run_bandit_episode
def run_bandit_episode(true_values, n_steps, epsilon, rng):
    k = len(true_values)
    q_values = np.zeros(k, dtype=float)
    action_counts = np.zeros(k, dtype=int)

    rewards = []
    actions = []

    for _ in range(n_steps):
        action = epsilon_greedy_action(q_values, epsilon, rng)
        reward = pull_arm(true_values, action, rng)

        q_values, action_counts = sample_average_update(
            q_values, action_counts, action, reward
        )

        actions.append(action)
        rewards.append(reward)

    return np.asarray(rewards), np.asarray(actions)

# Step 6 - track_rewards_and_optimal_actions
def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    rewards, actions = run_bandit_episode(
        true_values, n_steps, epsilon, rng
    )

    optimal_action = int(np.argmax(true_values))
    optimal_flags = (actions == optimal_action).astype(float)

    return rewards.astype(float), optimal_flags

# Step 7 - average_bandit_curves
def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    all_rewards = []
    all_optimal = []

    for i in range(n_runs):
        bandit = create_bandit_testbed(k, seed + i)
        rng = np.random.default_rng(seed + i)

        rewards, optimal = track_rewards_and_optimal_actions(
            bandit, n_steps, epsilon, rng
        )

        all_rewards.append(rewards)
        all_optimal.append(optimal)

    mean_reward = np.mean(all_rewards, axis=0)
    mean_optimal = np.mean(all_optimal, axis=0)

    return mean_reward, mean_optimal

# Step 8 - apply_random_walk_drift
def apply_random_walk_drift(true_values, drift_std, rng):
    noise = rng.normal(0, drift_std, size=true_values.shape)
    return true_values + noise

# Step 9 - constant_step_size_update
def constant_step_size_update(q_values, action, reward, alpha):
    q_values[action] += alpha * (reward - q_values[action])
    return q_values

# Step 10 - optimistic_initialization (not yet solved)
# TODO: implement

# Step 11 - ucb_action_select (not yet solved)
# TODO: implement

# Step 12 - gradient_bandit_update (not yet solved)
# TODO: implement

# Step 13 - bandit_parameter_study (not yet solved)
# TODO: implement

# Step 14 - build_gridworld_mdp (not yet solved)
# TODO: implement

# Step 15 - iterative_policy_evaluation (not yet solved)
# TODO: implement

# Step 16 - greedy_policy_improvement (not yet solved)
# TODO: implement

# Step 17 - policy_iteration (not yet solved)
# TODO: implement

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

