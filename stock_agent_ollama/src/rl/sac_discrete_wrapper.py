"""
Discrete to Continuous Action Space Wrapper for SAC

SAC requires continuous action space, but our trading environment uses discrete actions.
This wrapper converts the discrete 6-action space to continuous Box space and back.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteToBoxWrapper(gym.ActionWrapper):
    """
    Wrapper that converts a Discrete action space to a Box space for SAC.

    The continuous action is then discretized back to the original discrete actions:
    - Action 0: HOLD
    - Action 1: BUY_SMALL
    - Action 2: BUY_MEDIUM
    - Action 3: BUY_LARGE
    - Action 4: SELL_PARTIAL
    - Action 5: SELL_ALL

    Continuous range [-1, 1] is divided into n_discrete_actions bins.
    """

    def __init__(self, env, n_discrete_actions=6):
        """
        Args:
            env: The discrete action environment to wrap
            n_discrete_actions: Number of discrete actions (default: 6)
        """
        super().__init__(env)

        self.n_discrete_actions = n_discrete_actions

        # Replace discrete action space with continuous Box space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Define bin edges for discretization
        # Range [-1, 1] divided into n_discrete_actions equal bins
        self.bin_edges = np.linspace(-1.0, 1.0, n_discrete_actions + 1)

    def action(self, continuous_action):
        """
        Convert continuous action from SAC to discrete action for environment.

        Args:
            continuous_action: Float in range [-1, 1]

        Returns:
            discrete_action: Integer in range [0, n_discrete_actions-1]
        """
        # Ensure action is a scalar
        if isinstance(continuous_action, np.ndarray):
            continuous_action = continuous_action.item()

        # Clip to valid range
        continuous_action = np.clip(continuous_action, -1.0, 1.0)

        # Find which bin the action falls into
        # np.digitize returns bin index (1-indexed), subtract 1 for 0-indexing
        discrete_action = np.digitize(continuous_action, self.bin_edges[1:-1])

        # Ensure within bounds
        discrete_action = np.clip(discrete_action, 0, self.n_discrete_actions - 1)

        return discrete_action

    def step(self, action):
        """
        Takes a continuous action, discretizes it, and passes it to the wrapped environment.
        Also returns the discretized action in the info dict.
        """
        discrete_action = self.action(action)
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        info['discrete_action'] = discrete_action
        return obs, reward, terminated, truncated, info

    def reverse_action(self, discrete_action):
        """
        Convert discrete action to continuous (for analysis/visualization).

        Args:
            discrete_action: Integer in range [0, n_discrete_actions-1]

        Returns:
            continuous_action: Float in range [-1, 1]
        """
        # Map to center of the corresponding bin
        bin_width = 2.0 / self.n_discrete_actions
        continuous_action = -1.0 + (discrete_action + 0.5) * bin_width

        return continuous_action

    def get_diagnostics_summary(self):
        """
        Delegate to wrapped environment's diagnostics.

        Returns:
            Dictionary of diagnostics from the underlying environment
        """
        # Access the wrapped environment (ActionWrapper stores it as self.env)
        if hasattr(self.env, 'get_diagnostics_summary'):
            return self.env.get_diagnostics_summary()
        return {}

    def __getattr__(self, name):
        """
        Forward any unknown attributes to the wrapped environment.

        This allows SAC wrapper to expose all methods from EnhancedTradingEnv.
        """
        # Avoid infinite recursion by checking if env exists
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'env'")

        # Try to get attribute from wrapped environment
        return getattr(self.env, name)
