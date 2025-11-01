"""
A2C (Advantage Actor-Critic) agent for trading.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

from .base_agent import BaseRLAgent


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
            verbose: Verbosity level
            **kwargs: Additional A2C parameters
        """
        super().__init__(name="A2C")

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
