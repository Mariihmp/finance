# sac_agent.py (Corrected)
# sac_agent.py (Corrected Imports)
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

# CORRECTED: Import everything from the unified model.py
from model import ActorSAC, Critic, ReplayBuffer
# --- Hyperparameters ---
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
WEIGHT_DECAY = 0
ALPHA = 0.5
AUTO_ENTROPY = True
LR_ALPHA = 3e-4
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, fc1_units=24, fc2_units=48):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Alpha (Entropy Regularization)
        self.auto_entropy = AUTO_ENTROPY
        if self.auto_entropy:
            self.target_entropy = -torch.prod(torch.Tensor((action_size,)).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = ALPHA
            
        # Actor Network
        self.actor_local = ActorSAC(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Networks (Twin Q-Functions)
        self.critic_local_1 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target_1 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        self.critic_local_2 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target_2 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Hard copy weights to target networks
        self.soft_update(self.critic_local_1, self.critic_target_1, 1)
        self.soft_update(self.critic_local_2, self.critic_target_2, 1)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action, _ = self.actor_local.sample(state)
        self.actor_local.train()
        action_np = action.cpu().numpy()
        # <<< CORRECTION: Squeeze action to match environment's expected shape
        return np.clip((action_np + 1.0) / 2.0, 0, 1).squeeze()

    def reset(self):
        pass # Not needed for SAC

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions = (actions * 2.0) - 1.0 # Scale actions from [0,1] to [-1,1]

        # --- Update Critic ---
        with torch.no_grad():
            next_actions, next_log_prob = self.actor_local.sample(next_states)
            q_targets_next1 = self.critic_target_1(next_states, next_actions)
            q_targets_next2 = self.critic_target_2(next_states, next_actions)
            q_targets_next = torch.min(q_targets_next1, q_targets_next2) - self.alpha * next_log_prob
            q_targets = rewards + (gamma * q_targets_next * (1 - dones))
            
        q_expected1 = self.critic_local_1(states, actions)
        q_expected2 = self.critic_local_2(states, actions)
        critic_loss = F.mse_loss(q_expected1, q_targets) + F.mse_loss(q_expected2, q_targets)

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # --- Update Actor ---
        actions_pred, log_prob = self.actor_local.sample(states)
        q_pred1 = self.critic_local_1(states, actions_pred)
        q_pred2 = self.critic_local_2(states, actions_pred)
        q_pred = torch.min(q_pred1, q_pred2)
        actor_loss = (self.alpha * log_prob - q_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Alpha (Entropy Temperature) ---
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # --- Soft update target networks ---
        self.soft_update(self.critic_local_1, self.critic_target_1, TAU)
        self.soft_update(self.critic_local_2, self.critic_target_2, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)