# qlearning_agent.py
"""
Q-Learning agent for FrozenLake-v1
Saves:
 - q_table.npy
 - learning_curve.png
 - q_table_heatmap.png
 - training_rewards.csv
 - final_results.txt
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import csv

# ---------------------------
# Config / hyperparameters
# ---------------------------
ENV_NAME = "FrozenLake-v1"
NUM_EPISODES = 10000
LEARNING_RATE = 0.1    # alpha
DISCOUNT_FACTOR = 0.99 # gamma
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
SEED = 42
SAVE_DIR = "practical2_outputs"

# ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def choose_action(state, q_table, epsilon, env):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return int(np.argmax(q_table[state]))


def train_q_learning(env_name=ENV_NAME,
                     num_episodes=NUM_EPISODES,
                     learning_rate=LEARNING_RATE,
                     discount_factor=DISCOUNT_FACTOR,
                     epsilon_start=EPSILON_START,
                     epsilon_decay=EPSILON_DECAY,
                     epsilon_min=EPSILON_MIN,
                     seed=SEED):
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name, render_mode=None)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    epsilon = epsilon_start

    print("Training started:")
    for episode in range(num_episodes):
        state, info = env.reset(seed=None)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = choose_action(state, q_table, epsilon, env)
            next_state, reward, terminated, truncated, info = env.step(action)

            current_q = q_table[state, action]
            max_future_q = 0 if terminated else np.max(q_table[next_state])

            target_q = reward + discount_factor * max_future_q
            error = target_q - current_q
            new_q = current_q + learning_rate * error
            q_table[state, action] = new_q

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # print progress occasionally
        if (episode + 1) % 1000 == 0:
            avg_last_100 = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
            print(f"Episode {episode+1}/{num_episodes} - Avg reward (last100): {avg_last_100:.3f} - Epsilon: {epsilon:.3f}")

    env.close()
    results = {
        "q_table": q_table,
        "rewards_per_episode": rewards_per_episode,
        "params": {
            "env": env_name,
            "num_episodes": num_episodes,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon_start": epsilon_start,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "seed": seed
        }
    }
    return results


def plot_and_save(rewards, q_table, save_dir=SAVE_DIR):
    # save rewards csv
    rewards_csv = os.path.join(save_dir, "training_rewards.csv")
    with open(rewards_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i+1, r])

    # learning curve + moving average
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    window = 100
    moving_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(rewards[start:i+1]))
    plt.plot(moving_avg, linewidth=2, label=f"{window}-episode moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning Training Progress")
    plt.legend()

    # Q-table heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(q_table, aspect="auto")
    plt.colorbar()
    plt.xlabel("Action (0:Left,1:Down,2:Right,3:Up)")
    plt.ylabel("State")
    plt.title("Final Q-table")

    plt.tight_layout()
    learning_curve_path = os.path.join(save_dir, "learning_curve_and_qtable.png")
    plt.savefig(learning_curve_path)
    plt.close()
    print(f"Saved learning curve and q-table heatmap to {learning_curve_path}")

    # also save q_table as .npy
    qpath = os.path.join(save_dir, "q_table.npy")
    np.save(qpath, q_table)
    print(f"Saved Q-table to {qpath}")


def test_agent(q_table, env_name=ENV_NAME, num_episodes=1000, render=False):
    env_test = gym.make(env_name, render_mode="human" if render else None)
    wins = 0
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env_test.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

    env_test.close()
    success_rate = wins / num_episodes
    avg_reward = float(np.mean(total_rewards))
    return success_rate, avg_reward


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    results = train_q_learning()
    q_table = results["q_table"]
    rewards = results["rewards_per_episode"]

    plot_and_save(rewards, q_table)

    # test
    success_rate, avg_reward = test_agent(q_table, num_episodes=1000, render=False)
    res_text = (
        f"Tested Q-learning agent over 1000 episodes:\n"
        f"Success rate: {success_rate:.3%}\n"
        f"Average reward: {avg_reward:.4f}\n"
        f"Baseline random agent (approx): 6%\n"
        f"Improvement factor: {success_rate/0.06:.2f}x\n"
    )
    print(res_text)

    # save final results
    out_file = os.path.join(SAVE_DIR, "final_results.txt")
    with open(out_file, "w") as f:
        f.write(res_text)
    print(f"Wrote final results to {out_file}")
