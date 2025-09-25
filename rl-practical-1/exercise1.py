#CartPole
import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

print(f"Action Space: {env.action_space}")
#What type of space is the action space? How many actions are there?
#It's a discrete(2): There are two possible action(0:Left, 1:Right).

print(f"Observation Space: {env.observation_space}")
#What type of space is the observation space? The output is Box(4,). This represents a continuous space with 4 numbers. Based on the problem, what could these four numbers possibly represent?
# Observation Space: Box(4)
# This is a continuous space with 4 numbers.
# These 4 numbers represent:
# 1. Cart position (how far the cart is from the center)
# 2. Cart velocity (how fast the cart is moving left/right)
# 3. Pole angle (how tilted the pole is)
# 4. Pole angular velocity (how fast the pole is falling/rotating)

observation, info = env.reset()

terminated = False
truncated = False
total_reward = 0.0
while not terminated and not truncated:
    env.render()
    action = env.action_space.sample()
    print(f"Taking action: {action} (0:Left, 1:Right)") 
    
    next_observation, reward, terminated, truncated, info = env.step(action)

    #Run the random agent for one episode. What does the reward seem to represent in this environment? (Hint: you get a reward for every step the pole remains balanced).
    #reward represent that every time pole is balance reward=1.
    total_reward += reward
    observation = next_observation

    time.sleep(1)

print(f"\nEpisode finished! Total Reward: {total_reward}")
env.close()