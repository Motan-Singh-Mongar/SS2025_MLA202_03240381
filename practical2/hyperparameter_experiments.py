def analyze_q_table(q_table=None, q_table_path='practical2/q_table_final.npy'):
    """Analyze the trained Q-table"""
    if q_table is None:
        try:
            q_table = np.load(q_table_path)
            print("Loaded Q-table from file")
        except FileNotFoundError:
            print("Q-table file not found. Please run qlearning_agent.py first or provide a Q-table.")
            return
    
    print("\n=== Q-Table Analysis ===")
    
    # 1. Starting position analysis
    print(f"\n1. Starting Position (State 0) Q-values:")
    action_names = ['Left', 'Down', 'Right', 'Up']
    for action, name in enumerate(action_names):
        print(f"   {name}: {q_table[0, action]:.4f}")
    
    best_action = np.argmax(q_table[0])
    print(f"   Best action from start: {action_names[best_action]} (Q-value: {q_table[0, best_action]:.4f})")
    
    # 2. Find unexplored states
    unexplored_states = []
    for state in range(16):
        if np.all(q_table[state] == 0):
            unexplored_states.append(state)
    
    print(f"\n2. Exploration Analysis:")
    print(f"   Unexplored states: {unexplored_states}")
    print(f"   Percentage of unexplored states: {len(unexplored_states)/16:.1%}")
    
    # Count partially explored states (some actions tried)
    partially_explored = 0
    for state in range(16):
        if not np.all(q_table[state] == 0) and state not in unexplored_states:
            partially_explored += 1
    
    print(f"   Partially explored states: {partially_explored}")
    print(f"   Fully explored states: {16 - len(unexplored_states) - partially_explored}")
    
    # 3. Terminal state analysis
    hole_states = [5, 7, 11, 12]
    goal_state = 15
    
    print(f"\n3. Terminal State Analysis:")
    print(f"   Goal state (15) Q-values: {[f'{q:.4f}' for q in q_table[goal_state]]}")
    
    for hole in hole_states:
        hole_q_values = q_table[hole]
        print(f"   Hole state {hole} Q-values: {[f'{q:.4f}' for q in hole_q_values]}")
        if np.max(hole_q_values) > 0:
            print(f"     ⚠️  Warning: Positive Q-value found in hole state!")
    
    # 4. State with highest Q-value
    max_q_value = np.max(q_table)
    max_q_indices = np.where(q_table == max_q_value)
    
    print(f"\n4. Highest Q-values:")
    for i in range(len(max_q_indices[0])):
        state = max_q_indices[0][i]
        action = max_q_indices[1][i]
        print(f"   State {state}, Action {action_names[action]}: {max_q_value:.4f}")
    
    # 5. Policy visualization
    print(f"\n5. Learned Policy (best action per state):")
    policy = np.argmax(q_table, axis=1)
    policy_symbols = ['←', '↓', '→', '↑']
    
    print("   Grid visualization:")
    for i in range(0, 16, 4):
        row = []
        for j in range(4):
            state = i + j
            if state in hole_states:
                row.append('H')
            elif state == goal_state:
                row.append('G')
            else:
                row.append(policy_symbols[policy[state]])
        print("   " + " ".join(row))
    
    # 6. Confidence analysis
    print(f"\n6. Policy Confidence:")
    confidence_scores = []
    for state in range(16):
        if state not in hole_states + [goal_state]:
            q_values = q_table[state]
            if np.max(q_values) > 0:  # Only consider states with positive values
                best_q = np.max(q_values)
                second_best = np.partition(q_values, -2)[-2]
                confidence = best_q - second_best
                confidence_scores.append(confidence)
    
    if confidence_scores:
        avg_confidence = np.mean(confidence_scores)
        print(f"   Average confidence (Q-best - Q-second-best): {avg_confidence:.4f}")
    else:
        print("   Not enough data for confidence analysis")

def run_comprehensive_experiment():
    """Run one comprehensive experiment and analyze the resulting Q-table"""
    print("=== Running Comprehensive Q-Learning Experiment ===")
    
    env = gym.make("FrozenLake-v1")
    
    # Use best parameters from our experiments
    agent, rewards, success_rates, epsilon_history = train_q_learning_agent(
        env, 
        num_episodes=10000,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )
    
    # Test agent
    success_rate, avg_reward, avg_length = test_agent(env, agent, num_episodes=1000)
    print(f"\nFinal Performance: {success_rate:.1%} success rate")
    
    # Analyze the Q-table
    analyze_q_table(agent.q_table)
    
    # Save this Q-table for later use
    np.save('practical2/q_table_comprehensive.npy', agent.q_table)
    
    env.close()
    
    return agent, rewards, success_rates, epsilon_history

if __name__ == "__main__":
    # First, run a comprehensive experiment to generate a good Q-table
    agent, rewards, success_rates, epsilon_history = run_comprehensive_experiment()
    
    # Then run the hyperparameter comparisons (shorter runs for speed)
    print("\n" + "="*50)
    print("Starting Hyperparameter Comparisons...")
    print("="*50)
    
    lr_results = run_learning_rate_experiment()
    gamma_results = run_discount_factor_experiment()
    decay_results = run_epsilon_decay_experiment()
    
    # Plot comparison results
    plot_hyperparameter_results(lr_results, gamma_results, decay_results)
    
    # Answer key questions based on results
    print("\n=== Key Insights ===")
    print("1. Learning Rate (α):")
    best_lr = max(lr_results.items(), key=lambda x: x[1]['final_success'])
    worst_lr = min(lr_results.items(), key=lambda x: x[1]['final_success'])
    print(f"   Best: α = {best_lr[0]} ({best_lr[1]['final_success']:.1%} success)")
    print(f"   Worst: α = {worst_lr[0]} ({worst_lr[1]['final_success']:.1%} success)")
    
    print("\n2. Discount Factor (γ):")
    best_gamma = max(gamma_results.items(), key=lambda x: x[1]['final_success'])
    worst_gamma = min(gamma_results.items(), key=lambda x: x[1]['final_success'])
    print(f"   Best: γ = {best_gamma[0]} ({best_gamma[1]['final_success']:.1%} success)")
    print(f"   Worst: γ = {worst_gamma[0]} ({worst_gamma[1]['final_success']:.1%} success)")
    
    print("\n3. Epsilon Decay:")
    best_decay = max(decay_results.items(), key=lambda x: x[1]['final_success'])
    worst_decay = min(decay_results.items(), key=lambda x: x[1]['final_success'])
    print(f"   Best: decay = {best_decay[0]} ({best_decay[1]['final_success']:.1%} success)")
    print(f"   Worst: decay = {worst_decay[0]} ({worst_decay[1]['final_success']:.1%} success)")