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

# Step 4 - epsilon_greedy_action (not yet solved)
# TODO: implement

# Step 5 - run_bandit_episode (not yet solved)
# TODO: implement

# Step 6 - track_rewards_and_optimal_actions (not yet solved)
# TODO: implement

# Step 7 - average_bandit_curves (not yet solved)
# TODO: implement

# Step 8 - apply_random_walk_drift (not yet solved)
# TODO: implement

# Step 9 - constant_step_size_update (not yet solved)
# TODO: implement

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

