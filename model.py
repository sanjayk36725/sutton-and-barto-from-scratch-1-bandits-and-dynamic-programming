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

# Step 10 - optimistic_initialization
def optimistic_initialization(k, initial_value):
    return np.full(k, initial_value)

# Step 11 - ucb_action_select
def ucb_action_select(q_values, action_counts, timestep, c):
    # Any unvisited arm is preferred
    unvisited = np.where(action_counts == 0)[0]
    if len(unvisited) > 0:
        return int(unvisited[0])

    scores = q_values + c * np.sqrt(np.log(timestep) / action_counts)

    # np.argmax returns the smallest index when there is a tie
    return int(np.argmax(scores))

# Step 12 - gradient_bandit_update
def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    # Softmax policy
    exp_preferences = np.exp(preferences - np.max(preferences))
    policy = exp_preferences / np.sum(exp_preferences)

    advantage = reward - average_reward

    # Gradient-bandit update
    preferences = preferences.copy()
    preferences[action] += alpha * advantage * (1 - policy[action])

    for a in range(len(preferences)):
        if a != action:
            preferences[a] -= alpha * advantage * policy[a]

    return preferences

# Step 13 - bandit_parameter_study
def bandit_parameter_study(n_runs, n_steps, seed, settings):
    results = {}

    for setting in settings:
        method = setting["method"]
        param = setting["param"]
        nonstationary = setting.get("nonstationary", False)

        final_rewards = []

        for i in range(n_runs):
            rng = np.random.default_rng(seed + i)

            if nonstationary:
                true_values = np.zeros(10)
            else:
                true_values = create_bandit_testbed(10, seed + i)

            if method == "optimistic":
                q_values = optimistic_initialization(10, param)
            else:
                q_values = np.zeros(10)

            action_counts = np.zeros(10, dtype=int)
            preferences = np.zeros(10)
            avg_reward = 0.0

            rewards = []

            for t in range(n_steps):
                if method == "epsilon_greedy":
                    epsilon = param
                    if rng.random() < epsilon:
                        action = int(rng.integers(10))
                    else:
                        action = int(np.argmax(q_values))

                    reward = rng.normal(true_values[action], 1.0)
                    action_counts[action] += 1

                    n = action_counts[action]
                    q_values[action] += (reward - q_values[action]) / n

                elif method == "constant_step":
                    if rng.random() < 0.1:
                        action = int(rng.integers(10))
                    else:
                        action = int(np.argmax(q_values))

                    reward = rng.normal(true_values[action], 1.0)
                    action_counts[action] += 1

                    q_values = constant_step_size_update(
                        q_values, action, reward, param
                    )

                elif method == "optimistic":
                    action = int(np.argmax(q_values))

                    reward = rng.normal(true_values[action], 1.0)
                    action_counts[action] += 1

                    q_values = constant_step_size_update(
                        q_values, action, reward, 0.1
                    )

                elif method == "ucb":
                    unvisited = np.where(action_counts == 0)[0]

                    if len(unvisited) > 0:
                        action = int(unvisited[0])
                    else:
                        action = ucb_action_select(
                            q_values, action_counts, t + 1, param
                        )

                    reward = rng.normal(true_values[action], 1.0)
                    action_counts[action] += 1

                    n = action_counts[action]
                    q_values[action] += (reward - q_values[action]) / n

                elif method == "gradient":
                    exp_p = np.exp(preferences - np.max(preferences))
                    policy = exp_p / np.sum(exp_p)

                    action = int(rng.choice(10, p=policy))
                    reward = rng.normal(true_values[action], 1.0)

                    avg_reward = (
                        avg_reward * t + reward
                    ) / (t + 1)

                    preferences = gradient_bandit_update(
                        preferences,
                        action,
                        reward,
                        avg_reward,
                        param
                    )

                else:
                    raise ValueError("Unknown method")

                rewards.append(reward)

                if nonstationary:
                    true_values = apply_random_walk_drift(
                        true_values, 0.01, rng
                    )

            final_rewards.append(float(rewards[-1]))

        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"

        results[label] = float(np.mean(final_rewards))

    return results

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

