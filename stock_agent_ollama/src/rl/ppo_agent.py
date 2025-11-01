"""
PPO (Proximal Policy Optimization) agent for trading.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .base_agent import BaseRLAgent


class PPOAgent(BaseRLAgent):
    """
    PPO agent for trading strategies.

    PPO is a policy gradient method that is stable, sample-efficient,
    and works well for both discrete and continuous action spaces.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 0,
        **kwargs
    ):
        """
        Initialize PPO agent.

        Args:
            env: Trading environment
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            verbose: Verbosity level
            **kwargs: Additional PPO parameters
        """
        super().__init__(name="PPO")

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            **kwargs
        )

        self.training_stats = {}

    def train(
        self,
        env,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.

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
            'algorithm': 'PPO',
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
        """Save PPO model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path):
        """Load PPO model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = PPO.load(str(path))
        self.is_trained = True

    def get_info(self) -> Dict[str, Any]:
        """Get PPO agent information."""
        info = super().get_info()
        if self.model is not None:
            info.update({
                'learning_rate': self.model.learning_rate,
                'n_steps': self.model.n_steps,
                'batch_size': self.model.batch_size,
                'gamma': self.model.gamma,
            })
        return info
