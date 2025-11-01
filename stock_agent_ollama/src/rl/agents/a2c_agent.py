"""
A2C (Advantage Actor-Critic) agent for trading.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch.nn as nn

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .base_agent import BaseRLAgent
from ..networks import create_lstm_feature_extractor


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Wrapper to make LSTM feature extractor compatible with Stable-Baselines3.
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


class A2CAgent(BaseRLAgent):
    """
    A2C agent for trading strategies.

    A2C is an actor-critic method that is simpler and faster to train
    than PPO, but may be less stable. Good for quick experiments.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        normalize_advantage: bool = False,
        use_lstm: bool = False,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 2,
        features_dim: int = 128,
        verbose: int = 0,
        **kwargs
    ):
        """
        Initialize A2C agent.

        Args:
            env: Trading environment
            learning_rate: Learning rate
            n_steps: Steps per update
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            rms_prop_eps: RMSProp epsilon
            use_rms_prop: Whether to use RMSProp optimizer
            normalize_advantage: Whether to normalize advantages
            use_lstm: Whether to use LSTM feature extractor (hybrid architecture)
            lstm_hidden_size: LSTM hidden state size (if use_lstm=True)
            lstm_layers: Number of LSTM layers (if use_lstm=True)
            features_dim: Feature dimension output from extractor
            verbose: Verbosity level
            **kwargs: Additional A2C parameters
        """
        super().__init__(name="A2C-LSTM" if use_lstm else "A2C")

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
            # Configure actor-critic network architecture (post-LSTM)
            policy_kwargs['net_arch'] = [
                dict(pi=[128, 64], vf=[128, 64])  # Separate actor/critic networks
            ]

        self.model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            normalize_advantage=normalize_advantage,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            **kwargs
        )

        self.use_lstm = use_lstm
        self.training_stats = {}

    def train(
        self,
        env,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the A2C agent.

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

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )

        self.is_trained = True

        # Return training statistics
        return {
            'algorithm': 'A2C',
            'total_timesteps': total_timesteps,
            'status': 'completed'
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict action given observation.

        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, None)
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Agent must be trained or loaded before prediction")

        action, _states = self.model.predict(observation, deterministic=deterministic)
        return int(action), None

    def save(self, path: Path):
        """Save A2C model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path):
        """Load A2C model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = A2C.load(str(path))
        self.is_trained = True

    def get_info(self) -> Dict[str, Any]:
        """Get A2C agent information."""
        info = super().get_info()
        if self.model is not None:
            info.update({
                'learning_rate': self.model.learning_rate,
                'n_steps': self.model.n_steps,
                'gamma': self.model.gamma,
            })
        return info
