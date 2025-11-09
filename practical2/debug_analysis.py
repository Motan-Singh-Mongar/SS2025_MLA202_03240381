"""
Debug version - analyzes whatever Q-table is available
"""
import numpy as np
import os

def analyze_any_qtable():
    """Analyze any available Q-table file"""
    qtable_files = [
        'practical2/q_table_final.npy',
        'practical2/q_table_comprehensive.npy',
        'q_table_final.npy',
        'q_table_comprehensive.npy'
    ]
    
    q_table = None
    source_file = None
    
    for file in qtable_files:
        if os.path.exists(file):
            try:
                q_table = np.load(file)
                source_file = file
                print(f"✅ Loaded Q-table from: {file}")
                break
            except Exception as e:
                print(f"⚠️  Could not load {file}: {e}")
    
    if q_table is None:
        print("❌ No Q-table files found. Please run qlearning_agent.py first.")
        print("Creating a dummy Q-table for demonstration...")
        q_table = np.random.random((16, 4)) * 0.1
        source_file = "DEMO (random values)"
    
    # Perform analysis
    print(f"\nAnalyzing Q-table from: {source_file}")
    print(f"Q-table shape: {q_table.shape}")
    
    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"Min Q-value: {np.min(q_table):.4f}")
    print(f"Max Q-value: {np.max(q_table):.4f}")
    print(f"Mean Q-value: {np.mean(q_table):.4f}")
    print(f"Non-zero entries: {np.count_nonzero(q_table)}/{q_table.size}")
    
    # Policy
    policy = np.argmax(q_table, axis=1)
    action_names = ['Left', 'Down', 'Right', 'Up']
    policy_symbols = ['←', '↓', '→', '↑']
    
    print(f"\nLearned Policy:")
    for i in range(0, 16, 4):
        row = []
        for j in range(4):
            state = i + j
            row.append(f"{policy_symbols[policy[state]]}")
        print("  " + " ".join(row))
    
    print(f"\nBest actions per state:")
    for state in range(16):
        best_action = policy[state]
        best_q = q_table[state, best_action]
        print(f"State {state:2d}: {action_names[best_action]:5} (Q={best_q:.4f})")

if __name__ == "__main__":
    analyze_any_qtable()