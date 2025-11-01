"""
Base agent interface for RL trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path


class BaseRLAgent(ABC):
    """Base class for RL trading agents."""

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name/identifier
        """
        self.name = name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        env,
        total_timesteps: int,
        callback=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            env: Training environment
            total_timesteps: Number of timesteps to train
            callback: Optional training callback
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training statistics
        """
        pass

    @abstractmethod
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
            Tuple of (action, action_probabilities)
        """
        pass

    @abstractmethod
    def save(self, path: Path):
        """
        Save agent model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Path):
        """
        Load agent model from disk.

        Args:
            path: Path to load model from
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information.

        Returns:
            Dictionary with agent metadata
        """
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'algorithm': self.__class__.__name__
        }
