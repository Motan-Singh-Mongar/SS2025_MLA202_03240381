import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", render_mode="human")

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

num_episodes = 1000
episode_number = 0
rewards_per_episode = []
for episode in range(num_episodes): #Outer loop
    episode_number+=1

    observation, info = env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0
    while not terminated and not truncated: #Inner loop
        # ... take a random step ...
        # ... update episode_reward ...
   
        action = env.action_space.sample()
        f"Taking action: {action} (0:L, 1:D, 2:R, 3:U)"

        next_observation, reward, terminated, truncated, info = env.step(action)

        total_reward +=reward
        observation = next_observation
        rewards_per_episode.append(total_reward)
        # Append the final reward for this episode to the list
    
    print(f"\nEpisode:{episode_number} \nReward: {total_reward}\n")
average_reward = sum(rewards_per_episode) / num_episodes 
print('\nEpisode finished!\nTotal_reward:',sum(rewards_per_episode))
print(f"Average reward over {num_episodes} episodes: {average_reward:.4f}")

env.close()





































































