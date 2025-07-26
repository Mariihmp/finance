# train.py (Corrected to handle the DDPG baseline)
import numpy as np
import pandas as pd
from collections import deque
import argparse

import syntheticChrissAlmgren as sca
from model import ReplayBuffer 
from ddpg_agent import Agent as DDPGAgent
from td3_agent import Agent as TD3Agent
from sac_agent import Agent as SACAgent

def train(agent_type='DDPG', episodes=10000, risk_lambda=1e-6,
          random_seed=0,  
          lr_actor=1e-4, lr_critic=1e-3, policy_noise=0.2,
          fc1_units=24, fc2_units=48):

    env = sca.MarketEnvironment()
    state_size = env.observation_space_dimension()
    action_size = env.action_space_dimension()

    # --- THIS IS THE FIX ---
    # We check the agent type and initialize it with the correct parameters.
    if agent_type == 'DDPG':
        # For DDPG, we don't pass the extra hyperparameters
        agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=random_seed)
    else:
        # For TD3 and SAC, we create the full parameter dictionary
        agent_params = {
            'state_size': state_size, 'action_size': action_size, 'random_seed': random_seed,
            'lr_actor': lr_actor, 'lr_critic': lr_critic,
            'fc1_units': fc1_units, 'fc2_units': fc2_units
        }
        if agent_type == 'TD3':
            agent_params['policy_noise'] = policy_noise
            agent = TD3Agent(**agent_params)
        elif agent_type == 'SAC':
            agent = SACAgent(**agent_params)
        else:
            raise ValueError(f"Invalid agent type specified: {agent_type}")

    print(f"--- Training {agent_type} Agent with Lambda={risk_lambda} (Seed: {random_seed}) ---")

    shortfall_history = []
    shortfall_deque = deque(maxlen=100)

    for episode in range(1, episodes + 1):
        cur_state = env.reset(seed=episode, lamb=risk_lambda)
        agent.reset()
        env.start_transactions()

        for i in range(env.num_n + 1):
            action = agent.act(cur_state)
            new_state, reward, done, info = env.step(action)
            agent.step(cur_state, action, reward, new_state, done)
            cur_state = new_state
            if done:
                shortfall_history.append(info.implementation_shortfall)
                shortfall_deque.append(info.implementation_shortfall)
                break

        if episode % 100 == 0:
            avg_shortfall = np.mean(shortfall_deque) if len(shortfall_deque) > 0 else 0
            print(f'\rEpisode {episode}/{episodes}\tAverage Shortfall: ${avg_shortfall:,.2f}')

    print("\nTraining for this configuration has finished.")
    return shortfall_history

# ... (the rest of your train.py file for manual runs remains the same) ...