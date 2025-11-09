"""
Run all Q-learning experiments in correct order
"""
import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and check if it succeeds"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⚠️  {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    print("Starting Q-Learning Practical 2 Experiments")
    
    # Step 1: Run main Q-learning training
    if run_script("qlearning_agent.py"):
        print("✅ Q-learning training completed successfully")
    else:
        print("❌ Q-learning training failed")
        return
    
    # Step 2: Run hyperparameter experiments
    if run_script("hyperparameter_experiments.py"):
        print("✅ Hyperparameter experiments completed successfully")
    else:
        print("❌ Hyperparameter experiments failed")
        return
    
    # Step 3: Optional - Run demo
    demo = input("\nRun agent demonstration? (y/n): ").lower().strip()
    if demo == 'y':
        if run_script("test_agent.py"):
            print("✅ Agent demonstration completed successfully")
        else:
            print("❌ Agent demonstration failed")
    
    print("\n🎉 All experiments completed!")
    print("Check the 'images' folder for plots and analysis results.")

if __name__ == "__main__":
    main()