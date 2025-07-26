# multi_agent_env.py
import numpy as np

class MultiAgentMarketEnvironment:
    """
    A multi-agent version of the Almgren-Chriss market environment.

    This environment simulates a market where multiple RL agents trade simultaneously.
    The key changes are:
    - Market impact is based on the *sum* of all agents' trades.
    - Each agent receives a state that includes information about the other agents' actions.
    """

    def __init__(self,
                 num_agents=2,
                 T=1,
                 N=25,
                 X=1_000_000,
                 sigma=0.3,
                 s0=100.0,
                 eta=2.5e-6,
                 epsilon=2.5e-7):
        """
        Initializes the multi-agent environment.
        
        Args:
            num_agents (int): The number of RL agents in the simulation.
        """
        # Store parameters
        self.num_agents = num_agents
        self.T = T
        self.N = N
        self.dt = T / N
        self.X = X
        self.sigma = sigma
        self.s0 = s0
        self.eta = eta
        self.epsilon = epsilon
        
        # State variables that will be reset for each episode
        self.shares_remaining = [self.X] * self.num_agents
        self.s = self.s0
        self.n = 0
        self.lamb = 1e-6 # Default risk aversion, can be changed in reset()

    def observation_space_dimension(self):
        """The dimension of the state vector for a single agent."""
        # State: [time_fraction, inventory_fraction, others_volume_fraction]
        return 3

    def action_space_dimension(self):
        """The dimension of the action vector for a single agent."""
        return 1

    def reset(self, seed=None, lamb=1e-6):
        """Resets the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        self.lamb = lamb
        self.s = self.s0
        self.n = 0
        
        # Reset inventory for all agents
        # Inside the reset method
        self.shares_remaining = [self.X, int(self.X * 0.90)] # Agent 1 starts with 5% less
        
        # Initial state for each agent
        # The volume of other agents is 0 at the start.
        initial_state = [1.0, 1.0, 0.0] 
        
        # Return a list of identical initial states, one for each agent
        return [np.array(initial_state)] * self.num_agents

    def step(self, actions):
        """
        Executes one time step in the environment.
        
        Args:
            actions (list): A list of trading volumes, one for each agent.
        
        Returns:
            tuple: A tuple containing (next_states, rewards, dones, info).
                   Each element is a list corresponding to each agent.
        """
        self.n += 1
        
        # --- Core Multi-Agent Logic ---
        # 1. Sanitize actions: ensure they don't sell more than they have
        actions = [min(a, r) for a, r in zip(actions, self.shares_remaining)]
        actions = [max(a, 0) for a in actions]
        
        # 2. Calculate total volume from all agents
        total_volume = sum(actions)
        
        # 3. Calculate market impact based on total volume
        old_price = self.s
        price_drift = self.sigma * np.sqrt(self.dt) * np.random.randn()
        permanent_impact = self.eta * total_volume
        
        # Update price
        self.s = self.s - permanent_impact + price_drift
        
        # --- Calculate results for each agent individually ---
        next_states = []
        rewards = []
        dones = []
        
        for i in range(self.num_agents):
            agent_action = actions[i]
            
            # Temporary impact affects only this agent's execution price
            temporary_impact = self.epsilon * agent_action
            execution_price = old_price - temporary_impact
            
            # Calculate capture and shortfall for this agent
            capture = agent_action * execution_price
            
            # Update this agent's inventory
            self.shares_remaining[i] -= agent_action
            
            # Calculate reward for this agent
            # Reward = Captured value - Risk penalty
            reward = capture - self.lamb * (self.shares_remaining[i]**2)
            
            # Check if this agent is done (or if the episode is over)
            is_done = (self.n >= self.N)
            
            # Generate next state for this agent
            time_fraction = (self.N - self.n) / self.N
            inventory_fraction = self.shares_remaining[i] / self.X
            # The "other" volume is the total volume minus this agent's own action
            others_volume_fraction = (total_volume - agent_action) / self.X
            
            next_state = np.array([time_fraction, inventory_fraction, others_volume_fraction])
            
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(is_done)

        # Info dictionary can hold any extra data, e.g., for logging
        info = {'price': self.s}

        return next_states, rewards, dones, info