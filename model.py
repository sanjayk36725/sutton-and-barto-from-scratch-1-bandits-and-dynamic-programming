"""Reference implementations for Sutton & Barto bandits and dynamic programming."""

from __future__ import annotations

import numpy as np


MAX_SIMULATION_STEPS = 100_000
MAX_BANDIT_ARMS = 10_000
MAX_GRID_CELLS = 10_000
MAX_GAMBLER_GOAL = 10_000


def _validate_size(value: int, name: str, maximum: int) -> int:
    """Validate an integer size before using it in a NumPy allocation."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def create_bandit_testbed(k: int, seed: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    k = _validate_size(k, "k", MAX_BANDIT_ARMS)
    if std < 0:
        raise ValueError("std must be non-negative")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=k)


def pull_arm(true_values: np.ndarray, action: int, rng) -> float:
    if action < 0 or action >= len(true_values):
        raise IndexError("action out of range")
    return float(true_values[action] + rng.normal())


def sample_average_update(q_values, action_counts, action: int, reward: float):
    q_new = np.asarray(q_values, dtype=float).copy()
    c_new = np.asarray(action_counts, dtype=int).copy()
    c_new[action] += 1
    q_new[action] += (reward - q_new[action]) / c_new[action]
    return q_new, c_new


def epsilon_greedy_action(q_values, epsilon: float, rng) -> int:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    q = np.asarray(q_values, dtype=float)
    if q.size == 0:
        raise ValueError("q_values must not be empty")
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    best = np.flatnonzero(np.isclose(q, np.max(q)))
    return int(rng.choice(best))


def run_bandit_episode(true_values, n_steps: int, epsilon: float, rng, initial_q: float = 0.0):
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_steps > MAX_SIMULATION_STEPS:
        raise ValueError(f"n_steps must be <= {MAX_SIMULATION_STEPS}")
    k = _validate_size(len(true_values), "number of bandit arms", MAX_BANDIT_ARMS)
    q = np.full(k, float(initial_q))
    counts = np.zeros(k, dtype=int)
    rewards = np.empty(n_steps, dtype=float)
    for t in range(n_steps):
        action = epsilon_greedy_action(q, epsilon, rng)
        reward = pull_arm(np.asarray(true_values), action, rng)
        q, counts = sample_average_update(q, counts, action, reward)
        rewards[t] = reward
    return rewards


def track_rewards_and_optimal_actions(true_values, n_steps: int, epsilon: float, rng):
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_steps > MAX_SIMULATION_STEPS:
        raise ValueError(f"n_steps must be <= {MAX_SIMULATION_STEPS}")
    k = _validate_size(len(true_values), "number of bandit arms", MAX_BANDIT_ARMS)
    q = np.zeros(k, dtype=float)
    counts = np.zeros(k, dtype=int)
    rewards = np.empty(n_steps, dtype=float)
    optimal = np.empty(n_steps, dtype=float)
    optimal_action = int(np.argmax(true_values))
    for t in range(n_steps):
        action = epsilon_greedy_action(q, epsilon, rng)
        reward = pull_arm(np.asarray(true_values), action, rng)
        q, counts = sample_average_update(q, counts, action, reward)
        rewards[t] = reward
        optimal[t] = float(action == optimal_action)
    return rewards, optimal


def average_bandit_curves(k: int, n_runs: int, n_steps: int, epsilon: float, seed: int = 0):
    k = _validate_size(k, "k", MAX_BANDIT_ARMS)
    n_runs = _validate_size(n_runs, "n_runs", MAX_SIMULATION_STEPS)
    n_steps = _validate_size(n_steps, "n_steps", MAX_SIMULATION_STEPS)
    reward_sum = np.zeros(n_steps)
    optimal_sum = np.zeros(n_steps)
    for run in range(n_runs):
        true_values = create_bandit_testbed(k, seed + run)
        rng = np.random.default_rng(seed + 100_000 + run)
        rewards, optimal = track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng)
        reward_sum += rewards
        optimal_sum += optimal
    return reward_sum / n_runs, optimal_sum / n_runs


def apply_random_walk_drift(true_values, drift_std: float, rng):
    if drift_std < 0:
        raise ValueError("drift_std must be non-negative")
    values = np.asarray(true_values, dtype=float).copy()
    return values + rng.normal(0.0, drift_std, size=values.shape)


def constant_step_size_update(q_values, action: int, reward: float, alpha: float):
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    q = np.asarray(q_values, dtype=float).copy()
    q[action] += alpha * (reward - q[action])
    return q


def optimistic_initialization(k: int, initial_value: float = 5.0):
    k = _validate_size(k, "k", MAX_BANDIT_ARMS)
    return np.full(k, float(initial_value))


def ucb_action_select(q_values, action_counts, timestep: int, c: float = 2.0) -> int:
    q = np.asarray(q_values, dtype=float)
    counts = np.asarray(action_counts, dtype=float)
    unseen = np.flatnonzero(counts <= 0)
    if unseen.size:
        return int(unseen[0])
    t = max(int(timestep), 1)
    bonus = c * np.sqrt(np.log(t + 1.0) / counts)
    return int(np.argmax(q + bonus))


def gradient_bandit_update(preferences, action: int, reward: float, average_reward: float, alpha: float):
    h = np.asarray(preferences, dtype=float).copy()
    shifted = h - np.max(h)
    probs = np.exp(shifted)
    probs /= np.sum(probs)
    advantage = reward - average_reward
    h -= alpha * advantage * probs
    h[action] += alpha * advantage
    return h


def bandit_parameter_study(n_runs: int, n_steps: int, seed: int, settings):
    n_runs = _validate_size(n_runs, "n_runs", MAX_SIMULATION_STEPS)
    n_steps = _validate_size(n_steps, "n_steps", MAX_SIMULATION_STEPS)
    results = []
    for setting in settings:
        method = setting["method"]
        param = float(setting["param"])
        run_scores = []
        for run in range(n_runs):
            true_values = create_bandit_testbed(10, seed + run)
            rng = np.random.default_rng(seed + 10_000 + run)
            q = np.zeros(10)
            counts = np.zeros(10, dtype=int)
            prefs = np.zeros(10)
            avg_reward = 0.0
            rewards = []
            if method == "optimistic":
                q[:] = param
            for t in range(1, n_steps + 1):
                if method in {"epsilon_greedy", "optimistic"}:
                    eps = param if method == "epsilon_greedy" else 0.0
                    action = epsilon_greedy_action(q, eps, rng)
                elif method == "ucb":
                    action = ucb_action_select(q, counts, t, param)
                elif method == "gradient":
                    p = np.exp(prefs - np.max(prefs))
                    p /= p.sum()
                    action = int(rng.choice(len(p), p=p))
                else:
                    raise ValueError(f"unknown method: {method}")
                reward = pull_arm(true_values, action, rng)
                rewards.append(reward)
                counts[action] += 1
                if method == "gradient":
                    prefs = gradient_bandit_update(prefs, action, reward, avg_reward, param)
                    avg_reward += (reward - avg_reward) / t
                else:
                    q[action] += (reward - q[action]) / counts[action]
            run_scores.append(float(np.mean(rewards)))
        results.append({"method": method, "param": param, "average_reward": float(np.mean(run_scores))})
    return results


def build_gridworld_mdp(rows: int = 4, cols: int = 4):
    rows = _validate_size(rows, "rows", MAX_GRID_CELLS)
    cols = _validate_size(cols, "cols", MAX_GRID_CELLS)
    if rows * cols > MAX_GRID_CELLS:
        raise ValueError(f"grid must contain <= {MAX_GRID_CELLS} cells")
    n_states = rows * cols
    terminals = {0, n_states - 1}
    actions = [0, 1, 2, 3]  # up, right, down, left
    transitions = {}
    for s in range(n_states):
        transitions[s] = {}
        r, c = divmod(s, cols)
        for a in actions:
            if s in terminals:
                ns, reward, done = s, 0.0, True
            else:
                nr, nc = r, c
                if a == 0:
                    nr = max(0, r - 1)
                elif a == 1:
                    nc = min(cols - 1, c + 1)
                elif a == 2:
                    nr = min(rows - 1, r + 1)
                else:
                    nc = max(0, c - 1)
                ns = nr * cols + nc
                reward, done = -1.0, ns in terminals
            transitions[s][a] = [(1.0, ns, reward, done)]
    return {"rows": rows, "cols": cols, "n_states": n_states, "actions": actions, "terminals": terminals, "P": transitions}


def _action_value(mdp, state: int, action: int, values, gamma: float) -> float:
    total = 0.0
    for prob, ns, reward, done in mdp["P"][state][action]:
        total += prob * (reward + (0.0 if done else gamma * values[ns]))
    return float(total)


def iterative_policy_evaluation(mdp, policy=None, gamma: float = 1.0, theta: float = 1e-8):
    n = mdp["n_states"]
    actions = mdp["actions"]
    if policy is None:
        policy = np.full((n, len(actions)), 1.0 / len(actions))
    policy = np.asarray(policy, dtype=float)
    values = np.zeros(n)
    while True:
        delta = 0.0
        new_values = values.copy()
        for s in range(n):
            if s in mdp["terminals"]:
                continue
            v = sum(policy[s, i] * _action_value(mdp, s, a, values, gamma) for i, a in enumerate(actions))
            delta = max(delta, abs(v - values[s]))
            new_values[s] = v
        values = new_values
        if delta < theta:
            break
    return values


def greedy_policy_improvement(mdp, values, gamma: float = 1.0):
    policy = np.zeros(mdp["n_states"], dtype=int)
    for s in range(mdp["n_states"]):
        if s in mdp["terminals"]:
            policy[s] = 0
            continue
        qs = [_action_value(mdp, s, a, values, gamma) for a in mdp["actions"]]
        policy[s] = int(np.argmax(qs))
    return policy


def policy_iteration(mdp, gamma: float = 1.0, theta: float = 1e-8):
    n = mdp["n_states"]
    n_actions = len(mdp["actions"])
    deterministic = np.zeros(n, dtype=int)
    while True:
        probs = np.zeros((n, n_actions))
        probs[np.arange(n), deterministic] = 1.0
        values = iterative_policy_evaluation(mdp, probs, gamma, theta)
        improved = greedy_policy_improvement(mdp, values, gamma)
        if np.array_equal(improved, deterministic):
            return values, deterministic
        deterministic = improved


def value_iteration(mdp, gamma: float = 1.0, theta: float = 1e-8):
    values = np.zeros(mdp["n_states"])
    while True:
        delta = 0.0
        new_values = values.copy()
        for s in range(mdp["n_states"]):
            if s in mdp["terminals"]:
                continue
            best = max(_action_value(mdp, s, a, values, gamma) for a in mdp["actions"])
            delta = max(delta, abs(best - values[s]))
            new_values[s] = best
        values = new_values
        if delta < theta:
            break
    return values, greedy_policy_improvement(mdp, values, gamma)


def build_gambler_mdp(goal: int = 100, head_prob: float = 0.4):
    goal = _validate_size(goal, "goal", MAX_GAMBLER_GOAL)
    if goal <= 1:
        raise ValueError("goal must be greater than 1")
    if not 0.0 <= head_prob <= 1.0:
        raise ValueError("head_prob must be in [0, 1]")
    actions = {s: list(range(1, min(s, goal - s) + 1)) for s in range(1, goal)}
    return {"goal": goal, "head_prob": head_prob, "actions": actions}


def gambler_value_iteration(goal: int = 100, head_prob: float = 0.4, theta: float = 1e-9, gamma: float = 1.0):
    goal = _validate_size(goal, "goal", MAX_GAMBLER_GOAL)
    build_gambler_mdp(goal, head_prob)
    values = np.zeros(goal + 1)
    values[goal] = 1.0
    while True:
        delta = 0.0
        new_values = values.copy()
        for s in range(1, goal):
            stakes = range(1, min(s, goal - s) + 1)
            returns = [head_prob * gamma * values[s + a] + (1.0 - head_prob) * gamma * values[s - a] for a in stakes]
            best = max(returns) if returns else 0.0
            delta = max(delta, abs(best - values[s]))
            new_values[s] = best
        values = new_values
        if delta < theta:
            break
    return values


def extract_optimal_stakes(values, goal: int = 100, head_prob: float = 0.4, gamma: float = 1.0):
    goal = _validate_size(goal, "goal", MAX_GAMBLER_GOAL)
    build_gambler_mdp(goal, head_prob)
    values = np.asarray(values, dtype=float)
    if values.size != goal + 1:
        raise ValueError("values must contain goal + 1 entries")
    policy = np.zeros(goal + 1, dtype=int)
    for s in range(1, goal):
        stakes = np.arange(1, min(s, goal - s) + 1)
        if stakes.size == 0:
            continue
        returns = head_prob * gamma * values[s + stakes] + (1.0 - head_prob) * gamma * values[s - stakes]
        best = np.max(returns)
        policy[s] = int(stakes[np.flatnonzero(np.isclose(returns, best, atol=1e-12))[0]])
    return policy
