import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import train_q_learning_agent, test_agent

def run_learning_rate_experiment():
    print("=== Learning Rate Experiment ===")
    learning_rates = [0.01, 0.1, 0.5]
    results = {}
    
    for lr in learning_rates:
        print(f"Training with α = {lr}")
        env = gym.make("FrozenLake-v1")
        agent, rewards, success_rates, _ = train_q_learning_agent(
            env, num_episodes=1000, learning_rate=lr
        )
        success_rate = test_agent(agent, num_episodes=200)
        results[lr] = {'success_rates': success_rates, 'final_success': success_rate}
        env.close()
        print(f"α = {lr}: Success Rate = {success_rate:.1%}")
    
    return results

def run_discount_factor_experiment():
    print("\n=== Discount Factor Experiment ===")
    discount_factors = [0.9, 0.99, 0.999]
    results = {}
    
    for gamma in discount_factors:
        print(f"Training with γ = {gamma}")
        env = gym.make("FrozenLake-v1")
        agent, rewards, success_rates, _ = train_q_learning_agent(
            env, num_episodes=1000, discount_factor=gamma
        )
        success_rate = test_agent(agent, num_episodes=200)
        results[gamma] = {'success_rates': success_rates, 'final_success': success_rate}
        env.close()
        print(f"γ = {gamma}: Success Rate = {success_rate:.1%}")
    
    return results

def plot_results(lr_results, gamma_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Learning rates
    for lr, result in lr_results.items():
        ax1.plot(result['success_rates'], label=f'α={lr}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Learning Rate Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Discount factors
    for gamma, result in gamma_results.items():
        ax2.plot(result['success_rates'], label=f'γ={gamma}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Discount Factor Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('practical2/images/hyperparameter_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    lr_results = run_learning_rate_experiment()
    gamma_results = run_discount_factor_experiment()
    plot_results(lr_results, gamma_results)
    
    print("\n=== Best Parameters ===")
    best_lr = max(lr_results.items(), key=lambda x: x[1]['final_success'])
    best_gamma = max(gamma_results.items(), key=lambda x: x[1]['final_success'])
    print(f"Best learning rate: α = {best_lr[0]} ({best_lr[1]['final_success']:.1%})")
    print(f"Best discount factor: γ = {best_gamma[0]} ({best_gamma[1]['final_success']:.1%})")