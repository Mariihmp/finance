# td3_agent.py (Corrected)
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

# CORRECTED: Import everything from the unified model.py
from model import Actor, Critic, OUNoise, ReplayBuffer

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
WEIGHT_DECAY = 0
NOISE_CLIP = 0.5
POLICY_DELAY = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    # UPDATE THE __init__ SIGNATURE to accept tunable parameters
    def __init__(self, state_size, action_size, random_seed, 
                 lr_actor=1e-4, lr_critic=1e-3, fc1_units=24, fc2_units=48, policy_noise=0.2):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.policy_noise = policy_noise

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Networks
        self.critic_local_1 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target_1 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        self.critic_local_2 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target_2 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)
        
        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.update_counter = 0

        self.soft_update(self.critic_local_1, self.critic_target_1, 1)
        self.soft_update(self.critic_local_2, self.critic_target_2, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip((action.squeeze() + 1.0) / 2.0, 0, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        self.update_counter += 1
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-NOISE_CLIP, NOISE_CLIP)
            actions_next = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            Q_targets_next_1 = self.critic_target_1(next_states, actions_next)
            Q_targets_next_2 = self.critic_target_2(next_states, actions_next)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected_1 = self.critic_local_1(states, (actions * 2.0) - 1.0)
        Q_expected_2 = self.critic_local_2(states, (actions * 2.0) - 1.0)
        
        critic_loss_1 = F.mse_loss(Q_expected_1, Q_targets)
        critic_loss_2 = F.mse_loss(Q_expected_2, Q_targets)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()
        
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if self.update_counter % POLICY_DELAY == 0:
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.soft_update(self.critic_local_1, self.critic_target_1, TAU)
            self.soft_update(self.critic_local_2, self.critic_target_2, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local, target, tau):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)