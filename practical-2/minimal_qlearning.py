"""
Minimal Q-learning implementation to isolate issues
"""
import gymnasium as gym
import numpy as np
import random

# Create environment
env = gym.make("FrozenLake-v1")
print("Environment created")

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print("Q-table initialized")

# Training parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

# Train for a few episodes
print("Starting minimal training...")
for episode in range(100):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Q-learning update
        current_q = q_table[state, action]
        if terminated:
            max_future_q = 0
        else:
            max_future_q = np.max(q_table[next_state])
        
        target_q = reward + discount_factor * max_future_q
        q_table[state, action] = current_q + learning_rate * (target_q - current_q)
        
        state = next_state
        total_reward += reward
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (episode + 1) % 20 == 0:
        print(f"Episode {episode + 1}, Reward: {total_reward}")

# Save Q-table
np.save('practical2/q_table_minimal.npy', q_table)
print("Minimal Q-table saved!")

env.close()
print("Minimal training completed successfully!")