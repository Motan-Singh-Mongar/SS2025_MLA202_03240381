import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Create environment (no rendering during training)

env = gym.make("FrozenLake-v1", render_mode=None)  # Avoid pygame

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(f"Q-table shape: {q_table.shape}")
print(f"Initial Q-table (first 5 states):\n{q_table[:5]}")


# Epsilon-greedy action selection

def choose_action(state, q_table, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit


# Q-learning hyperparameters

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

# Training parameters
num_episodes = 10000
rewards_per_episode = []


# Training loop

print("Starting Q-learning training...")
for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = choose_action(state, q_table, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        current_q = q_table[state, action]
        max_future_q = 0 if terminated else np.max(q_table[next_state])
        target_q = reward + discount_factor * max_future_q
        q_table[state, action] = current_q + learning_rate * (target_q - current_q)

        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"Episode {episode+1}/{num_episodes}, Average Reward: {avg_reward:.3f}, Epsilon: {epsilon:.3f}")

print("Training completed!")

# Plot training result

plt.figure(figsize=(12, 4))

# Plot 1: Rewards with moving average

plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode, alpha=0.3, label='Episode Reward')
window_size = 100
moving_avg = [np.mean(rewards_per_episode[max(0, i-window_size):i+1]) for i in range(len(rewards_per_episode))]
plt.plot(moving_avg, label='Moving Average', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q-Learning Training Progress')
plt.legend()

# Plot 2: Final Q-table heatmap

plt.subplot(1, 2, 2)
plt.imshow(q_table, cmap='viridis', aspect='auto')
plt.xlabel('Actions (0:Left, 1:Down, 2:Right, 3:Up)')
plt.ylabel('States')
plt.title('Final Q-Table Values')
plt.colorbar()
plt.tight_layout()
plt.show()

# Test the trained agent
def test_agent(q_table, num_episodes=1000, render=False):
    """Test the trained Q-learning agent"""
    env_test = gym.make("FrozenLake-v1", render_mode="human" if render else None)
    wins = 0
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env_test.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = np.argmax(q_table[state])  # Pure exploitation
            state, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward

            if render and episode < 3:  # Only render first 3 episodes
                env_test.render()
                time.sleep(1)

        total_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

    env_test.close()
    success_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    return success_rate, avg_reward

# Run test

success_rate, avg_reward = test_agent(q_table, num_episodes=1000)
print(f"\nQ-Learning Agent Results (1000 episodes):")
print(f"Success Rate: {success_rate:.1%}")
print(f"Average Reward: {avg_reward:.4f}")
print(f"\nComparison to Random Agent (~6% success rate):")
print(f"Improvement: {success_rate/0.06:.1f}x better!")