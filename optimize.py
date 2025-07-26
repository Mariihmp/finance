# optimize.py (Corrected and Final Version)
import optuna
import argparse
import numpy as np
import train  # Imports the train function from your train.py

def objective(trial, agent_type, episodes, risk_lambda):
    """
    Objective function for Optuna to minimize.
    """
    
    # Define hyperparameter search space
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True)
    fc1 = trial.suggest_categorical("fc1_units", [16, 24, 32, 64, 128])
    fc2 = trial.suggest_categorical("fc2_units", [32, 48, 64, 128, 256])
    
    policy_noise = 0.0 # Default value
    if agent_type == 'TD3':
        policy_noise = trial.suggest_float("policy_noise", 0.1, 0.4)

    print(f"\n--- Starting Trial {trial.number} for {agent_type} ---")
    print(f"Parameters: LR_Actor={lr_actor:.6f}, LR_Critic={lr_critic:.6f}, FC1={fc1}, FC2={fc2}, Policy_Noise={policy_noise:.2f}")

    # =================== THIS IS THE CORRECTED BLOCK ===================
    # 1. Run training and get the full history of shortfalls
    full_history = train.train(
        agent_type=agent_type,
        episodes=episodes,
        risk_lambda=risk_lambda,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        policy_noise=policy_noise,
        fc1_units=fc1,
        fc2_units=fc2
    )

    # 2. Calculate the score to be optimized (average of last 100 episodes)
    if len(full_history) < 100:
        # Handle cases where training might fail early
        score = np.mean(full_history) if full_history else float('inf')
    else:
        score = np.mean(full_history[-100:])

    return score
    # =================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for DRL agents.")
    parser.add_argument("--agent", default="TD3", choices=["TD3", "SAC"], help="Agent to tune")
    parser.add_argument("--n_trials", default=30, type=int, help="Number of optimization trials to run")
    parser.add_argument("--episodes_per_trial", default=3000, type=int, help="Episodes per trial")
    parser.add_argument("--risk_lambda", default=1e-6, type=float, help="Fixed risk lambda for tuning")
    args = parser.parse_args()

    # --- THIS IS THE FIX ---
    # Check if the number of trials is positive before running.
    if args.n_trials <= 0:
        print(f"ERROR: Number of trials must be positive. You provided: {args.n_trials}")
    else:
        print(f"--- Starting new optimization study for {args.agent} with {args.n_trials} trials ---")
        
        study_name = f"{args.agent}-study"
        storage_name = f"sqlite:///{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize"
        )
        
        study.optimize(
            lambda trial: objective(trial, args.agent, args.episodes_per_trial, args.risk_lambda),
            n_trials=args.n_trials
        )

        print("\n--- Optimization Finished ---")
        if study.best_trial:
            print(f"Best trial for {args.agent}:")
            trial = study.best_trial
            print(f"  Value (min shortfall): ${trial.value:,.2f}")
            print("  Best Parameters: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
        else:
            print("No trials were completed.")