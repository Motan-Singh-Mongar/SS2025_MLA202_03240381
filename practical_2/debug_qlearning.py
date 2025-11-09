"""
Debug version of Q-learning to understand what's happening
"""

import gymnasium as gym
import numpy as np
import random

def choose_action(state, q_table, epsilon, env):
    """Choose action using epsilon-greedy strategy"""
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def main():
    """Debug Q-learning implementation"""
    env = gym.make("FrozenLake-v1", render_mode=None)

    print(f"Environment info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Let's check what happens in a few episodes
    print("\n=== Testing Random Episodes ===")
    random_rewards = []
    for episode in range(10):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        print(f"\nEpisode {episode + 1}: Starting at state {state}")

        while not terminated and not truncated and steps < 100:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if steps <= 5 or reward > 0 or terminated:  # Show first few steps and important events
                print(f"  Step {steps}: state {state} -> action {action} -> state {next_state}, reward {reward}")

            state = next_state

        print(f"  Episode ended: steps={steps}, total_reward={total_reward}, terminated={terminated}")
        random_rewards.append(total_reward)

    success_rate = sum(1 for r in random_rewards if r > 0) / len(random_rewards)
    print(f"\nRandom agent success rate (10 episodes): {success_rate:.1%}")

    # Now test Q-learning with debugging
    print("\n=== Testing Q-Learning ===")
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99  # Faster decay for debugging
    min_epsilon = 0.1

    rewards_per_episode = []
    successful_episodes = 0

    for episode in range(500):  # Shorter for debugging
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated and steps < 100:
            action = choose_action(state, q_table, epsilon, env)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Q-learning update
            current_q = q_table[state, action]

            if terminated:
                max_future_q = 0
            else:
                max_future_q = np.max(q_table[next_state])

            target_q = reward + discount_factor * max_future_q
            error = target_q - current_q
            new_q = current_q + learning_rate * error
            q_table[state, action] = new_q

            # Debug successful episodes
            if reward > 0:
                print(f"SUCCESS! Episode {episode}: state {state} -> action {action} -> reward {reward}")
                print(f"  Q-update: {current_q:.3f} -> {new_q:.3f}")
                successful_episodes += 1

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            recent_success = sum(1 for r in rewards_per_episode[-100:] if r > 0)
            print(f"Episode {episode + 1}: Recent success rate = {recent_success}%, epsilon = {epsilon:.3f}")

    print(f"\nTotal successful episodes during training: {successful_episodes}")
    print(f"Q-table after training:")
    print(f"Non-zero Q-values count: {np.count_nonzero(q_table)}")

    if np.count_nonzero(q_table) > 0:
        print("Some learned Q-values:")
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                if q_table[state, action] != 0:
                    print(f"  Q({state}, {action}) = {q_table[state, action]:.3f}")

    env.close()

if __name__ == "__main__":
    main()