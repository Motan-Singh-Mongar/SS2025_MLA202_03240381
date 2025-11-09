import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import deque
import os

# Create images folder if it doesn't exist
os.makedirs('practical2/images', exist_ok=True)

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros([state_size, action_size])
        
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state, terminated):
        """Q-learning update rule"""
        current_q = self.q_table[state, action]
        
        if terminated:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state])
            
        # Q-learning formula
        target_q = reward + self.discount_factor * max_future_q
        error = target_q - current_q
        new_q = current_q + self.learning_rate * error
        
        self.q_table[state, action] = new_q
        
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Reduce exploration over time"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

def train_q_learning_agent(env, num_episodes=10000, learning_rate=0.1, discount_factor=0.99, 
                          epsilon_start=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    """Train a Q-learning agent on FrozenLake"""
    
    # Initialize agent
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start
    )
    
    rewards_per_episode = []
    success_rate_history = []
    epsilon_history = []
    
    print("Starting Q-learning training...")
    print(f"Parameters: α={learning_rate}, γ={discount_factor}, ε_start={epsilon_start}")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state, terminated)
            
            # Move to next state
            state = next_state
            total_reward += reward
        
        # Store results
        rewards_per_episode.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        # Calculate success rate (last 100 episodes)
        if episode >= 100:
            recent_success_rate = np.mean(rewards_per_episode[-100:])
            success_rate_history.append(recent_success_rate)
        else:
            success_rate_history.append(0)
        
        # Decay epsilon
        agent.decay_epsilon(epsilon_decay, min_epsilon)
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_per_episode, success_rate_history, epsilon_history

def plot_training_results(rewards, success_rates, epsilon_history, q_table):
    """Plot training progress and Q-table"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rewards over time
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
    window_size = 100
    moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
    axes[0, 0].plot(moving_avg, label='Moving Average (100 episodes)', linewidth=2, color='red')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Q-Learning Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success rate over time
    axes[0, 1].plot(success_rates, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate (last 100 episodes)')
    axes[0, 1].set_title('Learning Progress - Success Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    axes[1, 0].plot(epsilon_history, linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate Decay')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Q-table heatmap
    im = axes[1, 1].imshow(q_table, cmap='viridis', aspect='auto')
    axes[1, 1].set_xlabel('Actions (0:Left, 1:Down, 2:Right, 3:Up)')
    axes[1, 1].set_ylabel('States')
    axes[1, 1].set_title('Final Q-Table Values')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('practical2/images/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_agent(env, agent, num_episodes=1000, render_first_3=False):
    """Test the trained agent"""
    if render_first_3:
        env_test = gym.make("FrozenLake-v1", render_mode="human")
    else:
        env_test = gym.make("FrozenLake-v1")
    
    wins = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, info = env_test.reset()
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not terminated and not truncated:
            # Pure exploitation (epsilon = 0)
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward
            steps += 1
            
            if render_first_3 and episode < 3:
                env_test.render()
                time.sleep(0.5)
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        if total_reward > 0:
            wins += 1
    
    env_test.close()
    
    success_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    return success_rate, avg_reward, avg_length

if __name__ == "__main__":
    # Create environment
    env = gym.make("FrozenLake-v1")
    
    # Train agent with default parameters
    print("=== Training Q-Learning Agent ===")
    agent, rewards, success_rates, epsilon_history = train_q_learning_agent(
        env, 
        num_episodes=10000,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )
    
    # Plot results
    plot_training_results(rewards, success_rates, epsilon_history, agent.q_table)
    
    # Test agent
    print("\n=== Testing Trained Agent ===")
    success_rate, avg_reward, avg_length = test_agent(env, agent, num_episodes=1000)
    
    print(f"Results after 1000 test episodes:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    
    # Compare to random baseline
    random_baseline = 0.06  # From practical1
    improvement = success_rate / random_baseline
    print(f"Improvement over random agent: {improvement:.1f}x")
    
    # Save Q-table for analysis - CORRECTED LINE
    np.save('practical2/q_table_final.npy', agent.q_table)  # Fixed: agent.q_table not agent.qt_table
    print("\nFinal Q-table saved to 'q_table_final.npy'")
    
    env.close()