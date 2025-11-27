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
        # Custom bins to center HOLD (Action 0) around 0.0
        # Mapping:
        # [-1.0, -0.6] -> SELL_ALL (5)
        # [-0.6, -0.2] -> SELL_PARTIAL (4)
        # [-0.2,  0.2] -> HOLD (0)
        # [ 0.2,  0.5] -> BUY_SMALL (1)
        # [ 0.5,  0.8] -> BUY_MEDIUM (2)
        # [ 0.8,  1.0] -> BUY_LARGE (3)
        self.bin_edges = np.array([-1.0, -0.6, -0.2, 0.2, 0.5, 0.8, 1.0])
        
        # Map bin index to discrete action
        # bin 0: [-1.0, -0.6] -> SELL_ALL (5)
        # bin 1: [-0.6, -0.2] -> SELL_PARTIAL (4)
        # bin 2: [-0.2,  0.2] -> HOLD (0)
        # bin 3: [ 0.2,  0.5] -> BUY_SMALL (1)
        # bin 4: [ 0.5,  0.8] -> BUY_MEDIUM (2)
        # bin 5: [ 0.8,  1.0] -> BUY_LARGE (3)
        self.bin_map = {
            0: 5, # SELL_ALL
            1: 4, # SELL_PARTIAL
            2: 0, # HOLD
            3: 1, # BUY_SMALL
            4: 2, # BUY_MEDIUM
            5: 3  # BUY_LARGE
        }

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
        # np.digitize returns bin index (1-indexed)
        # bins: [-1.0, -0.6, -0.2, 0.2, 0.5, 0.8, 1.0]
        # val: -0.8 -> index 1 (bin 0)
        bin_idx = np.digitize(continuous_action, self.bin_edges) - 1
        
        # Clamp index
        bin_idx = max(0, min(bin_idx, len(self.bin_map) - 1))

        return self.bin_map[bin_idx]

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
        Convert discrete action to continuous (approximate center of bin).
        """
        # Reverse map
        action_to_bin = {v: k for k, v in self.bin_map.items()}
        
        if discrete_action in action_to_bin:
            bin_idx = action_to_bin[discrete_action]
            low = self.bin_edges[bin_idx]
            high = self.bin_edges[bin_idx + 1]
            return (low + high) / 2.0
            
        return 0.0  # Default to HOLD center

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
