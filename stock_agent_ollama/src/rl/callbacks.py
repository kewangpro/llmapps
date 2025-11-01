"""
Training callbacks for monitoring and progress tracking.
"""

import numpy as np
from typing import Dict, List, Callable, Optional
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import json
import time


class TrainingProgressCallback(BaseCallback):
    """
    Callback for tracking training progress and metrics.
    """

    def __init__(
        self,
        check_freq: int = 1000,
        save_freq: int = 10000,
        save_path: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
        verbose: int = 0
    ):
        """
        Initialize training progress callback.

        Args:
            check_freq: Frequency to check and log metrics
            save_freq: Frequency to save model checkpoints
            save_path: Path to save checkpoints
            progress_callback: Optional callback function for UI updates
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.save_path = Path(save_path) if save_path else None
        self.progress_callback = progress_callback

        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        # Training statistics
        self.training_stats = {
            'timesteps': [],
            'mean_rewards': [],
            'std_rewards': [],
            'episode_lengths': [],
            'losses': [],
        }

        # Timing
        self.start_time = None
        self.last_check_time = None

    def _init_callback(self) -> None:
        """Initialize callback at training start."""
        self.start_time = time.time()
        self.last_check_time = self.start_time

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at each step.

        Returns:
            True to continue training, False to stop
        """
        # Track episode progress
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1

        # Check if episode is done
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        # Periodic logging
        if self.n_calls % self.check_freq == 0:
            self._log_progress()

        # Periodic saving
        if self.save_path and self.n_calls % self.save_freq == 0:
            self._save_checkpoint()

        return True

    def _log_progress(self):
        """Log training progress."""
        if len(self.episode_rewards) == 0:
            return

        # Calculate statistics
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        mean_length = np.mean(self.episode_lengths[-100:])

        # Update training stats
        self.training_stats['timesteps'].append(self.num_timesteps)
        self.training_stats['mean_rewards'].append(mean_reward)
        self.training_stats['std_rewards'].append(std_reward)
        self.training_stats['episode_lengths'].append(mean_length)

        # Calculate time metrics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        steps_per_sec = self.num_timesteps / elapsed_time

        # Log to console
        if self.verbose > 0:
            print(f"\nTimestep: {self.num_timesteps}")
            print(f"Episodes: {len(self.episode_rewards)}")
            print(f"Mean Reward (100 ep): {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Mean Episode Length: {mean_length:.1f}")
            print(f"Steps/sec: {steps_per_sec:.1f}")
            print(f"Elapsed: {elapsed_time:.1f}s")

        # Call progress callback for UI updates
        if self.progress_callback:
            progress_data = {
                'timestep': self.num_timesteps,
                'episodes': len(self.episode_rewards),
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_length': mean_length,
                'elapsed_time': elapsed_time,
                'steps_per_sec': steps_per_sec
            }
            self.progress_callback(progress_data)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        if self.save_path:
            checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(checkpoint_path)

            if self.verbose > 0:
                print(f"Saved checkpoint to {checkpoint_path}")

    def get_training_stats(self) -> Dict:
        """Get all training statistics."""
        return {
            **self.training_stats,
            'total_episodes': len(self.episode_rewards),
            'total_timesteps': self.num_timesteps,
            'all_episode_rewards': self.episode_rewards,
            'all_episode_lengths': self.episode_lengths
        }


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on reward threshold.
    """

    def __init__(
        self,
        reward_threshold: float,
        check_freq: int = 1000,
        min_episodes: int = 10,
        verbose: int = 0
    ):
        """
        Initialize early stopping callback.

        Args:
            reward_threshold: Reward threshold for early stopping
            check_freq: Frequency to check stopping condition
            min_episodes: Minimum episodes before checking
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        """
        Check early stopping condition.

        Returns:
            False to stop training, True to continue
        """
        # Track episodes
        self.current_episode_reward += self.locals.get('rewards', [0])[0]

        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        # Check stopping condition
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= self.min_episodes:
            mean_reward = np.mean(self.episode_rewards[-100:])

            if mean_reward >= self.reward_threshold:
                if self.verbose > 0:
                    print(f"\nEarly stopping: Mean reward {mean_reward:.2f} >= {self.reward_threshold:.2f}")
                return False

        return True


class PerformanceMonitorCallback(BaseCallback):
    """
    Callback to monitor and save best performing models.
    """

    def __init__(
        self,
        save_path: Path,
        check_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Initialize performance monitor callback.

        Args:
            save_path: Path to save best model
            check_freq: Frequency to check performance
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0

    def _init_callback(self) -> None:
        """Initialize callback."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Monitor performance and save best model.

        Returns:
            True to continue training
        """
        # Track episodes
        self.current_episode_reward += self.locals.get('rewards', [0])[0]

        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        # Check performance
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards[-100:])

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)

                if self.verbose > 0:
                    print(f"\nNew best mean reward: {mean_reward:.2f} - Model saved to {self.save_path}")

        return True


def create_training_callbacks(
    save_path: Optional[Path] = None,
    progress_callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    verbose: int = 1
) -> List[BaseCallback]:
    """
    Create a list of training callbacks.

    Args:
        save_path: Path to save checkpoints and best model
        progress_callback: Optional callback for UI updates
        reward_threshold: Optional reward threshold for early stopping
        verbose: Verbosity level

    Returns:
        List of callbacks
    """
    callbacks = []

    # Progress tracking
    callbacks.append(TrainingProgressCallback(
        check_freq=1000,
        save_freq=10000,
        save_path=save_path / "checkpoints" if save_path else None,
        progress_callback=progress_callback,
        verbose=verbose
    ))

    # Best model saving
    if save_path:
        callbacks.append(PerformanceMonitorCallback(
            save_path=save_path / "best_model.zip",
            check_freq=1000,
            verbose=verbose
        ))

    # Early stopping
    if reward_threshold is not None:
        callbacks.append(EarlyStoppingCallback(
            reward_threshold=reward_threshold,
            check_freq=1000,
            min_episodes=20,
            verbose=verbose
        ))

    return callbacks
