import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", render_mode="human")

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

# --- THE MAIN LOOP ---

# 1. RESET the environment
# This function is called at the beginning of every new episode.
# It returns the initial observation and some optional info.
observation, info = env.reset()

# Initialize variables to track the episode's progress
terminated = False
truncated = False
total_reward = 0.0


# The loop continues as long as the episode is not finished.

while not terminated and not truncated:
    # 2. RENDER the environment (optional)
    # This displays the current state in the popup window.
    env.render()

    # 3. CHOOSE an action
    # For now, we select a completely random action from the action space.
    action = env.action_space.sample()
    print(f"Taking action: {action} (0:L, 1:D, 2:R, 3:U)")

    # 4. STEP the environment
    # This is the most important function. It executes the chosen action.
    # It returns five crucial pieces of information:
    #   next_observation: The agent's new location.
    #   reward: The reward obtained from the last action. For FrozenLake, this is 1.0 if we reach the goal 'G', and 0.0 otherwise.
    #   terminated: A boolean. True if the episode ended because of a terminal state (reaching 'G' or falling in 'H').
    #   truncated: A boolean. True if the episode ended for another reason (e.g., a time limit was reached).
    #   info: A dictionary for debugging info, which we can ignore for now.
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Update our tracking variables
    total_reward += reward
    observation = next_observation

    # Add a small delay to make the visualization easier to follow
    time.sleep(0.5)

# After the loop, the episode is finished.
print(f"\nEpisode finished! Total Reward: {total_reward}")

# 5. CLOSE the environment
# This is important for cleaning up resources, especially the rendering window.
env.close()