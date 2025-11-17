"""
DQN (Deep Q-Network) agent for trading with action masking support.

DQN is particularly well-suited for stock trading because:
- Sample efficient (experience replay)
- Learns from rare profitable trades
- Integrates action masking easily (mask Q-values before argmax)
- Deterministic execution (no sampling randomness)
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch
import torch.nn as nn

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .base_agent import BaseRLAgent
from ..networks import create_lstm_feature_extractor


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Wrapper to make LSTM feature extractor compatible with Stable-Baselines3 DQN.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 2,
        use_attention: bool = False
    ):
        super().__init__(observation_space, features_dim)

        self.lstm_extractor = create_lstm_feature_extractor(
            observation_space=observation_space,
            features_dim=features_dim,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            use_attention=use_attention
        )

    def forward(self, observations):
        return self.lstm_extractor(observations)


class DQNAgent(BaseRLAgent):
    """
    DQN agent for trading strategies with action masking.

    DQN advantages for trading:
    - Experience replay: learns from rare profitable trades repeatedly
    - Sample efficiency: better for limited historical data
    - Action masking: easily masks invalid actions by setting Q-values to -inf
    - Deterministic: always selects argmax(Q) during inference
    """

    def __init__(
        self,
        env,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 10000,
        batch_size: int = 128,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        use_lstm: bool = False,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 2,
        features_dim: int = 128,
        verbose: int = 0,
        **kwargs
    ):
        """
        Initialize DQN agent optimized for stock trading.

        Args:
            env: Trading environment
            learning_rate: Learning rate (lower than PPO for stability)
            buffer_size: Replay buffer size (stores experiences)
            learning_starts: Steps before learning begins (fill buffer)
            batch_size: Minibatch size for updates
            tau: Soft target network update coefficient
            gamma: Discount factor
            train_freq: Update frequency (steps between updates)
            gradient_steps: Gradient steps per update
            target_update_interval: Steps between target network updates
            exploration_fraction: Fraction of training for exploration (eps decay)
            exploration_initial_eps: Initial epsilon for epsilon-greedy
            exploration_final_eps: Final epsilon for epsilon-greedy
            use_lstm: Whether to use LSTM feature extractor
            lstm_hidden_size: LSTM hidden state size (if use_lstm=True)
            lstm_layers: Number of LSTM layers (if use_lstm=True)
            features_dim: Feature dimension output from extractor
            verbose: Verbosity level
            **kwargs: Additional DQN parameters
        """
        super().__init__(name="DQN-LSTM" if use_lstm else "DQN")

        # Configure policy kwargs for LSTM feature extractor
        policy_kwargs = kwargs.pop('policy_kwargs', {})

        if use_lstm:
            # Use custom LSTM feature extractor
            policy_kwargs['features_extractor_class'] = LSTMFeaturesExtractor
            policy_kwargs['features_extractor_kwargs'] = {
                'features_dim': features_dim,
                'lstm_hidden_size': lstm_hidden_size,
                'num_lstm_layers': lstm_layers,
                'use_attention': False
            }
            # Configure Q-network architecture (post-LSTM)
            policy_kwargs['net_arch'] = [128, 64]
        else:
            # Standard MLP architecture for Q-network
            policy_kwargs['net_arch'] = [256, 128, 64]

        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            **kwargs
        )

        self.use_lstm = use_lstm
        self.training_stats = {}
        self.action_masking_enabled = False

        # Store environment to access action masking
        self.env = env

    def train(
        self,
        env,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the DQN agent.

        Args:
            env: Training environment
            total_timesteps: Number of timesteps to train
            callback: Optional training callback
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training statistics
        """
        # Update environment if different
        if env != self.model.env:
            self.model.set_env(env)
            self.env = env

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )

        self.is_trained = True

        # Return training statistics
        return {
            'algorithm': 'DQN',
            'total_timesteps': total_timesteps,
            'status': 'completed'
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict action given observation with optional action masking.

        This is the key innovation for DQN in trading: we can apply action masks
        by setting Q-values of invalid actions to -inf before argmax.

        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy (ignored, DQN is always deterministic during inference)
            action_mask: Optional binary mask (1=valid, 0=invalid). If provided, invalid actions get Q=-inf

        Returns:
            Tuple of (action, None)
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Agent must be trained or loaded before prediction")

        # Get Q-values for all actions
        obs_tensor = torch.as_tensor(observation).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            q_values = self.model.q_net(obs_tensor).cpu().numpy()[0]

        # Apply action masking if provided
        if action_mask is not None:
            # Set Q-values of invalid actions to -infinity
            masked_q = q_values.copy()
            masked_q[action_mask == 0] = -np.inf

            # Select action with highest Q-value among valid actions
            action = int(np.argmax(masked_q))
        else:
            # Standard DQN: select action with highest Q-value
            action = int(np.argmax(q_values))

        return action, None

    def predict_with_env_mask(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict action using action mask from environment.

        This method fetches the action mask from the environment and applies it.
        Use this during backtesting and live trading for safe action selection.

        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, None)
        """
        # Get action mask from environment if available
        action_mask = None
        if hasattr(self.env, 'get_action_mask'):
            action_mask = self.env.get_action_mask()

        return self.predict(observation, deterministic, action_mask)

    def save(self, path: Path):
        """Save DQN model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path, env=None):
        """
        Load DQN model from disk.

        Args:
            path: Path to model file
            env: Optional environment (needed to use action masking)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = DQN.load(str(path))
        self.is_trained = True

        if env is not None:
            self.env = env

    def get_info(self) -> Dict[str, Any]:
        """Get DQN agent information."""
        info = super().get_info()
        if self.model is not None:
            info.update({
                'learning_rate': self.model.learning_rate,
                'buffer_size': self.model.buffer_size,
                'batch_size': self.model.batch_size,
                'gamma': self.model.gamma,
                'exploration_rate': getattr(self.model, 'exploration_rate', 0.0),
            })
        return info
