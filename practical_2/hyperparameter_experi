"""
Hyperparameter Experiments for Q-Learning Agent
Exercise 1: Investigating how hyperparameters affect learning

This script tests different values for:
1. Learning Rate (α) - 0.01, 0.1, 0.5
2. Discount Factor (γ) - 0.9, 0.99, 0.999
3. Epsilon Decay - 0.99 (fast) vs 0.999 (slow)
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import defaultdict

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

def train_q_learning(learning_rate, discount_factor, epsilon_decay, num_episodes=5000):
    """Train Q-learning agent with specified hyperparameters"""
    env = gym.make("FrozenLake-v1", render_mode=None)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    epsilon = 1.0
    min_epsilon = 0.01
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(state, q_table, epsilon, env)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Q-learning update rule
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

    env.close()

    # Test final performance
    success_rate, avg_reward = test_agent(q_table, num_episodes=1000)

    return q_table, rewards_per_episode, success_rate, avg_reward

def experiment_learning_rate():
    """Experiment 1: Test different learning rates"""
    print("=== Learning Rate Experiment ===")
    learning_rates = [0.01, 0.1, 0.5]
    results = {}

    for lr in learning_rates:
        print(f"Training with learning rate α = {lr}")
        q_table, rewards, success_rate, avg_reward = train_q_learning(
            learning_rate=lr,
            discount_factor=0.99,
            epsilon_decay=0.995
        )
        results[lr] = {
            'rewards': rewards,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'q_table': q_table
        }
        print(f"  Final success rate: {success_rate:.1%}")

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for lr in learning_rates:
        # Calculate moving average
        rewards = results[lr]['rewards']
        window_size = 100
        moving_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window_size)
            moving_avg.append(np.mean(rewards[start:i+1]))

        plt.plot(moving_avg, label=f'α = {lr}')

    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Learning Rate Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    lrs = list(results.keys())
    success_rates = [results[lr]['success_rate'] for lr in lrs]

    plt.bar([str(lr) for lr in lrs], success_rates)
    plt.xlabel('Learning Rate (α)')
    plt.ylabel('Final Success Rate')
    plt.title('Final Performance by Learning Rate')

    plt.tight_layout()
    plt.savefig('practical2/learning_rate_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

def experiment_discount_factor():
    """Experiment 2: Test different discount factors"""
    print("\n=== Discount Factor Experiment ===")
    discount_factors = [0.9, 0.99, 0.999]
    results = {}

    for gamma in discount_factors:
        print(f"Training with discount factor γ = {gamma}")
        q_table, rewards, success_rate, avg_reward = train_q_learning(
            learning_rate=0.1,
            discount_factor=gamma,
            epsilon_decay=0.995
        )
        results[gamma] = {
            'rewards': rewards,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'q_table': q_table
        }
        print(f"  Final success rate: {success_rate:.1%}")

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for gamma in discount_factors:
        rewards = results[gamma]['rewards']
        window_size = 100
        moving_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window_size)
            moving_avg.append(np.mean(rewards[start:i+1]))

        plt.plot(moving_avg, label=f'γ = {gamma}')

    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Discount Factor Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    gammas = list(results.keys())
    success_rates = [results[gamma]['success_rate'] for gamma in gammas]

    plt.bar([str(gamma) for gamma in gammas], success_rates)
    plt.xlabel('Discount Factor (γ)')
    plt.ylabel('Final Success Rate')
    plt.title('Final Performance by Discount Factor')

    plt.tight_layout()
    plt.savefig('practical2/discount_factor_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

def experiment_epsilon_decay():
    """Experiment 3: Test different epsilon decay rates"""
    print("\n=== Epsilon Decay Experiment ===")
    epsilon_decays = [0.99, 0.999]
    decay_names = ['Fast Decay (0.99)', 'Slow Decay (0.999)']
    results = {}

    for decay, name in zip(epsilon_decays, decay_names):
        print(f"Training with epsilon decay = {decay} ({name})")
        q_table, rewards, success_rate, avg_reward = train_q_learning(
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon_decay=decay
        )
        results[name] = {
            'rewards': rewards,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'q_table': q_table,
            'decay': decay
        }
        print(f"  Final success rate: {success_rate:.1%}")

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for name in results.keys():
        rewards = results[name]['rewards']
        window_size = 100
        moving_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window_size)
            moving_avg.append(np.mean(rewards[start:i+1]))

        plt.plot(moving_avg, label=name)

    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Epsilon Decay Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    names = list(results.keys())
    success_rates = [results[name]['success_rate'] for name in names]

    plt.bar(names, success_rates)
    plt.xlabel('Epsilon Decay Strategy')
    plt.ylabel('Final Success Rate')
    plt.title('Final Performance by Epsilon Decay')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('practical2/epsilon_decay_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

def print_analysis():
    """Print analysis questions and expected insights"""
    print("\n=== Deep Understanding Questions ===")

    print("\n1. Learning Rate Analysis:")
    print("   - How does learning rate affect convergence speed and final performance?")
    print("   - Why might α = 0.01 learn slowly but be more stable?")
    print("   - What happens with α = 0.5 - does it learn faster or become unstable?")
    print("   - Key Insight: Learning rate controls update size - high=unstable, low=slow")

    print("\n2. Discount Factor Analysis:")
    print("   - Why might higher discount factor perform better on FrozenLake?")
    print("   - How does γ affect the agent's 'patience' for long-term rewards?")
    print("   - What happens with γ = 0.9 vs γ = 0.999 in final performance?")
    print("   - Key Insight: FrozenLake only rewards at goal - need high γ for distant reward")

    print("\n3. Epsilon Decay Analysis:")
    print("   - What happens if we stop exploring too quickly (fast decay)?")
    print("   - How does exploration schedule affect final performance?")
    print("   - Which decay rate leads to better learning and why?")
    print("   - Key Insight: Balance exploration (learning) vs exploitation (using knowledge)")

def main():
    """Run all hyperparameter experiments"""
    print("Starting Hyperparameter Experiments")
    print("This will test different values for learning rate, discount factor, and epsilon decay")
    print("Each experiment trains for 5000 episodes and tests final performance over 1000 episodes\n")

    # Run experiments
    lr_results = experiment_learning_rate()
    gamma_results = experiment_discount_factor()
    epsilon_results = experiment_epsilon_decay()

    # Summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)

    print("\nLearning Rate Results:")
    for lr, result in lr_results.items():
        print(f"  α = {lr:4}: Success Rate = {result['success_rate']:.1%}")

    print("\nDiscount Factor Results:")
    for gamma, result in gamma_results.items():
        print(f"  γ = {gamma:5}: Success Rate = {result['success_rate']:.1%}")

    print("\nEpsilon Decay Results:")
    for name, result in epsilon_results.items():
        print(f"  {name}: Success Rate = {result['success_rate']:.1%}")

    # Print analysis questions
    print_analysis()

    # Find best hyperparameters
    best_lr = max(lr_results.keys(), key=lambda x: lr_results[x]['success_rate'])
    best_gamma = max(gamma_results.keys(), key=lambda x: gamma_results[x]['success_rate'])

    print(f"\n=== BEST HYPERPARAMETERS ===")
    print(f"Best Learning Rate: α = {best_lr} (Success Rate: {lr_results[best_lr]['success_rate']:.1%})")
    print(f"Best Discount Factor: γ = {best_gamma} (Success Rate: {gamma_results[best_gamma]['success_rate']:.1%})")

    return lr_results, gamma_results, epsilon_results

if __name__ == "__main__":
    lr_results, gamma_results, epsilon_results = main()