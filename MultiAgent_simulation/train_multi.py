# train_multi.py
import numpy as np
import pandas as pd
from collections import deque
import sys
import os
import torch

# --- ROBUST PATH CORRECTION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from multi_agent_env import MultiAgentMarketEnvironment
from sac_agent import Agent as SACAgent 

def train(num_agents=2, episodes=3000, risk_lambda=1e-6):
    env = MultiAgentMarketEnvironment(num_agents=num_agents)
    state_size = env.observation_space_dimension()
    action_size = env.action_space_dimension()
    agents = [SACAgent(state_size=state_size, action_size=action_size, random_seed=i) for i in range(num_agents)]
    print(f"Initialized {len(agents)} SAC agents.")

    results = {f'agent_{i}': [] for i in range(num_agents)}
    shortfall_deques = [deque(maxlen=100) for _ in range(num_agents)]

    for i_episode in range(1, episodes + 1):
        states = env.reset(seed=i_episode, lamb=risk_lambda)
        for agent in agents:
            agent.reset()

        for t in range(env.N + 1):
            actions_unscaled = [agent.act(state) for agent, state in zip(agents, states)]
            actions = [a * env.X for a in actions_unscaled]
            
            next_states, rewards, dones, info = env.step(actions)

            for i, agent in enumerate(agents):
                agent.step(states[i], actions_unscaled[i], rewards[i], next_states[i], dones[i])
            
            states = next_states
            
            if all(dones):
                initial_value = env.X * env.s0
                for i in range(num_agents):
                    final_value = (env.X - env.shares_remaining[i]) * info['price']
                    shortfall = initial_value - final_value
                    results[f'agent_{i}'].append(shortfall)
                    shortfall_deques[i].append(shortfall)
                break
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}/{episodes}...', end="")
            for i in range(num_agents):
                avg_shortfall = np.mean(shortfall_deques[i])
                print(f' | Agent {i} Avg Shortfall: ${avg_shortfall:,.2f}', end="")
            print()

    return results, agents

if __name__ == '__main__':
    NUM_AGENTS = 2
    EPISODES = 3000

    print("--- Starting Multi-Agent Training ---")
    final_results, trained_agents = train(num_agents=NUM_AGENTS, episodes=EPISODES)
    print("\n--- Training Finished ---")
    
    for i, agent in enumerate(trained_agents):
        agent_key = f'agent_{i}'
        df = pd.DataFrame(final_results[agent_key], columns=['shortfall'])
        csv_filename = f'results_multi_{agent_key}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"✅ Results for {agent_key} saved to {csv_filename}")

        actor_filename = f'agent_{i}_actor.pth'
        torch.save(agent.actor_local.state_dict(), actor_filename)
        print(f"✅ Model for {agent_key} saved to {actor_filename}")