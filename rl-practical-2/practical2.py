# hyperparameter_experiments.py
"""
Run hyperparameter sweeps (learning rate, discount factor, epsilon decay)
Saves results to CSV and plots a comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from qlearning_agent import train_q_learning, test_agent, plot_and_save, SAVE_DIR

# experiment grid
learning_rates = [0.01, 0.1, 0.5]
discount_factors = [0.9, 0.99, 0.999]
epsilon_decays = [0.99, 0.995, 0.999]

out_dir = os.path.join(SAVE_DIR, "hyperparam_experiments")
os.makedirs(out_dir, exist_ok=True)

records = []

# small number of episodes per experiment to get quick signal; adjust higher for final runs
EPISODES_PER_EXP = 3000

for alpha in learning_rates:
    for gamma in discount_factors:
        for ed in epsilon_decays:
            start = time.time()
            print(f"Running alpha={alpha}, gamma={gamma}, eps_decay={ed}")
            results = train_q_learning(
                num_episodes=EPISODES_PER_EXP,
                learning_rate=alpha,
                discount_factor=gamma,
                epsilon_decay=ed,
                epsilon_start=1.0,
                epsilon_min=0.01,
            )
            q_table = results["q_table"]
            rewards = results["rewards_per_episode"]
            success_rate, avg_reward = test_agent(q_table, num_episodes=500, render=False)

            elapsed = time.time() - start
            record = {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon_decay": ed,
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "episodes_trained": EPISODES_PER_EXP,
                "elapsed_seconds": elapsed
            }
            print(f"Result: success_rate={success_rate:.3%}, avg_reward={avg_reward:.4f}, time={elapsed:.1f}s")
            records.append(record)

            # save q_table and learning curve for this run (optional)
            # create descriptive filename
            tag = f"a{alpha}_g{gamma}_ed{ed}".replace(".", "p")
            subfolder = os.path.join(out_dir, tag)
            os.makedirs(subfolder, exist_ok=True)
            np.save(os.path.join(subfolder, "q_table.npy"), q_table)

# save all results
df = pd.DataFrame(records)
csv_path = os.path.join(out_dir, "hyperparam_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved experiment results to {csv_path}")

# plot success_rate heatmap grouped by alpha/gamma for a fixed epsilon_decay (choose median)
sample_ed = epsilon_decays[1]  # 0.995
df_ed = df[df["epsilon_decay"] == sample_ed]

pivot = df_ed.pivot(index="alpha", columns="gamma", values="success_rate")
plt.figure(figsize=(6,4))
plt.imshow(pivot, aspect="auto")
plt.colorbar(label="success_rate")
plt.title(f"Success Rate (epsilon_decay={sample_ed})")
plt.ylabel("alpha")
plt.xlabel("gamma")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"heatmap_ed{sample_ed}.png"))
plt.close()
print("Hyperparameter sweep complete.")
