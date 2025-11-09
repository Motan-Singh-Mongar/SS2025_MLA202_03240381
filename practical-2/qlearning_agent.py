import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
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
        self.q_table = np.zeros([state_size, action_size])
        
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state, action]
        if terminated:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state])
            
        target_q = reward + self.discount_factor * max_future_q
        error = target_q - current_q
        new_q = current_q + self.learning_rate * error
        self.q_table[state, action] = new_q
        
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

def train_q_learning_agent(env, num_episodes=2000, learning_rate=0.1, discount_factor=0.99, 
                          epsilon_start=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    
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
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_q_value(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        if episode >= 100:
            recent_success_rate = np.mean(rewards_per_episode[-100:])
            success_rate_history.append(recent_success_rate)
        else:
            success_rate_history.append(0)
        
        agent.decay_epsilon(epsilon_decay, min_epsilon)
        
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_per_episode, success_rate_history, epsilon_history

def plot_training_results(rewards, success_rates, epsilon_history, q_table):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Rewards
    axes[0, 0].plot(rewards, alpha=0.3, color='blue')
    window_size = 100
    moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
    axes[0, 0].plot(moving_avg, linewidth=2, color='red')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success rate
    axes[0, 1].plot(success_rates, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Success Rate (last 100 episodes)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    axes[1, 0].plot(epsilon_history, linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Q-table heatmap
    im = axes[1, 1].imshow(q_table, cmap='viridis', aspect='auto')
    axes[1, 1].set_xlabel('Actions')
    axes[1, 1].set_ylabel('States')
    axes[1, 1].set_title('Q-Table Values')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('practical2/images/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_agent(agent, num_episodes=500):
    env = gym.make("FrozenLake-v1")
    wins = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            wins += 1
    
    env.close()
    success_rate = wins / num_episodes
    return success_rate

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    
    print("=== Training Q-Learning Agent ===")
    agent, rewards, success_rates, epsilon_history = train_q_learning_agent(
        env, 
        num_episodes=2000,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )
    
    plot_training_results(rewards, success_rates, epsilon_history, agent.q_table)
    
    print("\n=== Testing Trained Agent ===")
    success_rate = test_agent(agent, num_episodes=500)
    
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Improvement over random agent (6%): {success_rate/0.06:.1f}x")
    
    np.save('practical2/q_table_final.npy', agent.q_table)
    print("Q-table saved to 'q_table_final.npy'")
    
    env.close()