# risk_adjusted_env.py
import numpy as np
from collections import deque


class RiskAdjustedMarketEnvironment:
    """
    An environment where the reward is a risk-adjusted measure.
    Reward = Captured Value - lambda * Volatility
    """

    def __init__(self, T=1, N=50, X=1_000_000, sigma=0.3, s0=100.0,
                 eta=2.5e-6, epsilon=2.5e-7, volatility_window=10):
        # Store parameters
        self.T = T
        self.N = N
        self.dt = T / N
        self.X = X
        self.sigma = sigma
        self.s0 = s0
        self.eta = eta
        self.epsilon = epsilon

        # --- NEW: Parameters for volatility calculation ---
        self.volatility_window = volatility_window
        self.price_history = deque(maxlen=self.volatility_window + 1)

        # State variables
        self.shares_remaining = self.X
        self.s = self.s0
        self.n = 0
        self.lamb = 1e-6  # Default, can be changed in reset()

    def observation_space_dimension(self):
        # State: [time_fraction, inventory_fraction]
        return 2

    def action_space_dimension(self):
        return 1

    def reset(self, seed=None, lamb=1e-6):
        if seed is not None:
            np.random.seed(seed)

        self.lamb = lamb
        self.s = self.s0
        self.n = 0
        self.shares_remaining = self.X

        # --- NEW: Reset and pre-fill price history ---
        self.price_history.clear()
        for _ in range(self.volatility_window + 1):
            self.price_history.append(self.s0)

        initial_state = np.array([1.0, 1.0])
        return initial_state

    def step(self, action):
        # Action is a fraction from the agent, scale it to shares
        action = float(action)
        trade_volume = min(action * self.X, self.shares_remaining)
        trade_volume = max(trade_volume, 0)

        self.n += 1

        # Market impact calculations
        old_price = self.s
        price_drift = self.sigma * np.sqrt(self.dt) * np.random.randn()
        permanent_impact = self.eta * trade_volume
        self.s = self.s - permanent_impact + price_drift

        # Add new price to history
        self.price_history.append(self.s)

        # Calculate this step's results
        temporary_impact = self.epsilon * trade_volume
        execution_price = old_price - temporary_impact
        captured_value = trade_volume * execution_price

        self.shares_remaining -= trade_volume

        # --- NEW: Risk-Adjusted Reward Calculation ---
        prices = np.array(self.price_history)
        if len(prices) < 2:
            volatility = 0
        else:
            log_returns = np.log(prices[1:] / prices[:-1])
            volatility = np.std(log_returns)

        reward = captured_value - self.lamb * volatility

        # State and done signal
        is_done = (self.n >= self.N)
        time_fraction = (self.N - self.n) / self.N
        inventory_fraction = self.shares_remaining / self.X
        next_state = np.array([time_fraction, inventory_fraction])

        # Info dict for logging
        info = {'shortfall': (self.X * self.s0) -
                (captured_value + self.shares_remaining * self.s)}

        return next_state, reward, is_done, info
