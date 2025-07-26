# run_experiments.py
import train
import pandas as pd

# --- 1. DEFINE GLOBAL PARAMETERS ---
# Set the number of episodes, risk parameter, and random seeds for the experiments.
# You can add more seeds to test for robustness, e.g., [42, 123, 2025]
EPISODES = 10000
RISK_LAMBDA = 1e-6
SEEDS = [42] 

# --- 2. DEFINE YOUR EXPERIMENTS ---
# This is the main list of experiments to run.
# To add a new experiment, copy a dictionary and change its parameters.
# To skip an experiment, you can comment it out.
experiment_configs = [
    {
        'name': 'DDPG_baseline', 
        'agent_type': 'DDPG',
        # No specific hyperparameters needed; uses defaults from the agent file.
    },
    # {
    #     'name': 'TD3_best', 
    #     'agent_type': 'TD3',
    #     'lr_actor': 0.00045, 
    #     'lr_critic': 0.00088, 
    #     'fc1': 32, 
    #     'fc2': 128, 
    #     'policy_noise': 0.12 # Specific to TD3
    # },
    {
        'name': 'SAC_best', 
        'agent_type': 'SAC',
        'lr_actor': 0.0003, 
        'lr_critic': 0.0003, 
        'fc1': 64, 
        'fc2': 256
    },
    # Example of another experiment you could add later:
    # {
    #     'name': 'SAC_small_network', 
    #     'agent_type': 'SAC',
    #     'fc1': 32, 
    #     'fc2': 32
    # },
]

# --- 3. RUN THE EXPERIMENT LOOP ---
if __name__ == '__main__':
    for seed in SEEDS:
        print(f"\n##################################################")
        print(f"## RUNNING ALL EXPERIMENTS FOR SEED: {seed} ##")
        print(f"##################################################")
        
        for config in experiment_configs:
            print(f"\n" + "="*50)
            print(f"▶️ Starting Experiment: {config['name']} (Seed: {seed})")
            print("="*50)
            
            # This function call passes the hyperparameters from the config.
            # config.get('key', default_value) safely gets a value if it exists,
            # or uses a default if it doesn't (e.g., for DDPG_baseline).
            shortfall_history = train.train(
                agent_type=config['agent_type'],
                episodes=EPISODES,
                risk_lambda=RISK_LAMBDA,
                random_seed=seed,
                # Hyperparameters from config dict
                lr_actor=config.get('lr_actor', 1e-4), # Default DDPG/TD3 LR
                lr_critic=config.get('lr_critic', 1e-3), # Default DDPG/TD3 LR
                fc1_units=config.get('fc1', 24),
                fc2_units=config.get('fc2', 48),
                policy_noise=config.get('policy_noise', 0.2) # Default TD3 noise
            )
            
            # Save results to a CSV file named after the experiment and seed
            results_df = pd.DataFrame(shortfall_history, columns=['shortfall'])
            filename = f"results_{config['name']}_seed_{seed}.csv"
            results_df.to_csv(filename, index=False)
            print(f"✅ Experiment {config['name']} (Seed: {seed}) finished. Results saved to {filename}")