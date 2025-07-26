# run_experiments.py
import pandas as pd
import torch
import sys
from collections import deque
import numpy as np

# Import our new environment and the agents
from risk_adjusted_env import RiskAdjustedMarketEnvironment
from ddpg_agent import Agent as DDPGAgent
from sac_agent import Agent as SACAgent
from syntheticChrissAlmgren import MarketEnvironment


def train(agent, env, episodes=3000, risk_lambda=1e-6):
    shortfall_history = []
    shortfall_deque = deque(maxlen=100)
    for i_episode in range(1, episodes + 1):
        state = env.reset(seed=i_episode, lamb=risk_lambda)
        agent.reset()
        total_captured_value = 0

        for t in range(env.N + 1):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                shortfall = info['shortfall']
                shortfall_history.append(shortfall)
                shortfall_deque.append(shortfall)
                break

        if i_episode % 100 == 0:
            print(
                f'\rEpisode {i_episode}/{episodes} | Avg Shortfall: ${np.mean(shortfall_deque):,.2f}')

    return shortfall_history


if __name__ == '__main__':
    EPISODES = 3000

    experiment_configs = [
        {'name': 'DDPG_RiskAdjusted', 'agent_class': DDPGAgent},
        {'name': 'SAC_RiskAdjusted', 'agent_class': SACAgent},
    ]

    env = RiskAdjustedMarketEnvironment() ##
    state_size = env.observation_space_dimension()
    action_size = env.action_space_dimension()

    for config in experiment_configs:
        print("\n" + "="*50)
        print(f"Starting Experiment: {config['name']}")
        print("="*50)

        agent = config['agent_class'](
            state_size=state_size, action_size=action_size, random_seed=42)

        results = train(agent, env, episodes=EPISODES)

        # Save results and model
        df = pd.DataFrame(results, columns=['shortfall'])
        csv_filename = f"results_{config['name']}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

        model_filename = f"{config['name']}_actor.pth"
        torch.save(agent.actor_local.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
