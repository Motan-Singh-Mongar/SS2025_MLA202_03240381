"""
Quick test of Q-learning implementation with reduced training episodes
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def choose_action(state, q_table, epsilon, env):
    """Choose action using epsilon-greedy strategy"""
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def test_agent(q_table, num_episodes=1000):
    """Test the trained Q-learning agent"""
    env_test = gym.make("FrozenLake-v1", render_mode=None)

    wins = 0
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env_test.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

    env_test.close()
    success_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    return success_rate, avg_reward

def main():
    """Quick test of Q-learning with reduced episodes"""
    env = gym.make("FrozenLake-v1", render_mode=None)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    print(f"Q-table shape: {q_table.shape}")

    # Q-learning hyperparameters
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    # Reduced training for quick test
    num_episodes = 2000
    rewards_per_episode = []

    print("Starting Q-learning training (quick test - 2000 episodes)...")

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state, q_table, epsilon, env)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Q-learning update
            current_q = q_table[state, action]

            if terminated:
                max_future_q = 0
            else:
                max_future_q = np.max(q_table[next_state])

            target_q = reward + discount_factor * max_future_q
            error = target_q - current_q
            new_q = current_q + learning_rate * error
            q_table[state, action] = new_q

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.3f}, Epsilon: {epsilon:.3f}")

    print("Training completed!")

    # Test the agent
    print("Testing trained agent...")
    success_rate, avg_reward = test_agent(q_table, num_episodes=1000)

    print(f"\nQ-Learning Agent Results (1000 test episodes):")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"\nComparison to Random Agent (~6% success rate):")
    print(f"Improvement: {success_rate/0.06:.1f}x better!")

    # Quick Q-table analysis
    print(f"\nQ-Table Analysis:")
    print(f"State 0 (start) Q-values: {q_table[0]}")
    print(f"Best action from start: {np.argmax(q_table[0])} (0:Left, 1:Down, 2:Right, 3:Up)")

    env.close()

    return success_rate >= 0.7  # Return True if meets 70% requirement

if __name__ == "__main__":
    meets_requirement = main()
    print(f"\nMeets >70% requirement: {meets_requirement}")