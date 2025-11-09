import gymnasium as gym
import numpy as np
from qlearning_agent import QLearningAgent
import time

def demo_agent(episodes=3):
    """Demonstrate the trained agent"""
    env = gym.make("FrozenLake-v1", render_mode="human")
    
    # Load trained Q-table
    q_table = np.load('practical2/q_table_final.npy')
    agent = QLearningAgent(16, 4)
    agent.q_table = q_table
    agent.epsilon = 0  # Pure exploitation
    
    print("=== Agent Demonstration ===")
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not terminated and not truncated:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.5)
            
            if terminated:
                if reward > 0:
                    print(f"🎉 SUCCESS! Reached goal in {steps} steps")
                else:
                    print(f"💥 Fell in hole at step {steps}")
        
        print(f"Total reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    demo_agent(episodes=3)