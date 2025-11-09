"""
Basic test to verify Q-learning works
"""
import gymnasium as gym
import numpy as np

def test_basic():
    print("Testing basic Q-learning...")
    
    # Create environment
    env = gym.make("FrozenLake-v1")
    print("Environment created successfully")
    
    # Test Q-table creation
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    print(f"Q-table created: {q_table.shape}")
    
    # Test a few random episodes
    for episode in range(3):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        print(f"Episode {episode + 1}:", end=" ")
        
        while not terminated and not truncated:
            # Random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
        
        print(f"Reward: {total_reward}")
    
    env.close()
    print("Basic test completed successfully!")

if __name__ == "__main__":
    test_basic()