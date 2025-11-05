"""
Training orchestration for RL trading agents.

This module contains the trainer, callbacks, and reward functions.
"""

from .agents import create_agent, BaseRLAgent
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Callable, Optional, Any, TYPE_CHECKING
import json
import numpy as np
import time

if TYPE_CHECKING:
    from .environments import SingleStockTradingEnv



# ==================== REWARD FUNCTIONS ====================

@dataclass
class RewardConfig:
    return_weight: float = 1.0
    risk_penalty: float = 0.5
    sharpe_bonus: float = 0.1
    transaction_cost_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    max_drawdown_penalty: float = 0.3

    # Penalty for extreme actions
    extreme_action_penalty: float = 0.01

    # Bonus for profitable trades
    profitable_trade_bonus: float = 0.05


class RewardFunction:
    """Base reward function for trading strategies."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.prev_portfolio_value = None
        self.peak_portfolio_value = None
        self.returns_history = []

    def reset(self):
        """Reset reward function state."""
        self.prev_portfolio_value = None
        self.peak_portfolio_value = None
        self.returns_history = []

    def calculate(
        self,
        portfolio_value: float,
        action: int,
        prev_action: int,
        cash: float,
        position: float,
        price: float,
        prev_price: float,
        **kwargs
    ) -> float:
        """
        Calculate reward for the current step.

        Args:
            portfolio_value: Current total portfolio value
            action: Current action taken
            prev_action: Previous action
            cash: Current cash holdings
            position: Current stock position
            price: Current stock price
            prev_price: Previous stock price
            **kwargs: Additional metrics (volatility, etc.)

        Returns:
            Calculated reward value
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class SimpleReturnReward(RewardFunction):
    """Simple reward based on portfolio value change."""

    def calculate(
        self,
        portfolio_value: float,
        action: int,
        prev_action: int,
        cash: float,
        position: float,
        price: float,
        prev_price: float,
        **kwargs
    ) -> float:
        """Calculate simple return-based reward."""
        if self.prev_portfolio_value is None:
            self.prev_portfolio_value = portfolio_value
            return 0.0

        # Calculate return
        portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value

        # Apply transaction costs if action changed
        transaction_cost = 0.0
        if action != prev_action and action != 1:  # 1 is HOLD
            transaction_value = abs(action - prev_action) * price * position if position > 0 else price
            transaction_cost = transaction_value * self.config.transaction_cost_rate

        reward = portfolio_return - (transaction_cost / self.prev_portfolio_value if self.prev_portfolio_value > 0 else 0)

        self.prev_portfolio_value = portfolio_value
        return reward


class RiskAdjustedReward(RewardFunction):
    """Risk-adjusted reward incorporating Sharpe ratio and drawdown."""

    def __init__(self, config: Optional[RewardConfig] = None, window_size: int = 20):
        super().__init__(config)
        self.window_size = window_size

    def calculate(
        self,
        portfolio_value: float,
        action: int,
        prev_action: int,
        cash: float,
        position: float,
        price: float,
        prev_price: float,
        **kwargs
    ) -> float:
        """Calculate risk-adjusted reward."""
        if self.prev_portfolio_value is None:
            self.prev_portfolio_value = portfolio_value
            self.peak_portfolio_value = portfolio_value
            return 0.0

        # Calculate return
        portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.returns_history.append(portfolio_return)

        # Keep only recent history
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Base reward: portfolio return
        reward = portfolio_return * self.config.return_weight

        # Transaction costs
        transaction_cost = 0.0
        slippage_cost = 0.0

        if action != prev_action and action != 1:  # Action changed and not HOLD
            # Transaction cost
            transaction_value = price * position if position > 0 else price * 100  # Assume 100 shares default
            transaction_cost = transaction_value * self.config.transaction_cost_rate

            # Slippage (market impact)
            slippage_cost = transaction_value * self.config.slippage_rate

            # Extreme action penalty (discourage excessive trading)
            if abs(action - prev_action) > 2:
                reward -= self.config.extreme_action_penalty

        # Apply costs
        total_cost = (transaction_cost + slippage_cost) / self.prev_portfolio_value if self.prev_portfolio_value > 0 else 0
        reward -= total_cost

        # Risk penalty: volatility
        if len(self.returns_history) >= 2:
            volatility = np.std(self.returns_history)
            reward -= volatility * self.config.risk_penalty

            # Sharpe bonus (if we have enough data)
            if len(self.returns_history) >= 5:
                mean_return = np.mean(self.returns_history)
                sharpe = mean_return / (volatility + 1e-8)
                if sharpe > 0:
                    reward += sharpe * self.config.sharpe_bonus

        # Drawdown penalty
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value

        if drawdown > 0.1:  # More than 10% drawdown
            reward -= drawdown * self.config.max_drawdown_penalty

        # Profitable trade bonus
        if portfolio_return > 0:
            reward += self.config.profitable_trade_bonus

        self.prev_portfolio_value = portfolio_value
        return reward


class CustomizableReward(RewardFunction):
    """Customizable reward with adjustable weights."""

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        window_size: int = 20,
        use_sharpe: bool = True,
        use_drawdown: bool = True,
        use_transaction_costs: bool = True,
        use_slippage: bool = False
    ):
        super().__init__(config)
        self.window_size = window_size
        self.use_sharpe = use_sharpe
        self.use_drawdown = use_drawdown
        self.use_transaction_costs = use_transaction_costs
        self.use_slippage = use_slippage

    def calculate(
        self,
        portfolio_value: float,
        action: int,
        prev_action: int,
        cash: float,
        position: float,
        price: float,
        prev_price: float,
        **kwargs
    ) -> float:
        """Calculate customizable reward."""
        if self.prev_portfolio_value is None:
            self.prev_portfolio_value = portfolio_value
            self.peak_portfolio_value = portfolio_value
            return 0.0

        # Base return
        portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.returns_history.append(portfolio_return)

        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        reward = portfolio_return * self.config.return_weight

        # Optional: Transaction costs
        if self.use_transaction_costs and action != prev_action and action != 1:
            transaction_value = price * position if position > 0 else price * 100
            transaction_cost = transaction_value * self.config.transaction_cost_rate

            if self.use_slippage:
                slippage_cost = transaction_value * self.config.slippage_rate
                transaction_cost += slippage_cost

            reward -= transaction_cost / self.prev_portfolio_value if self.prev_portfolio_value > 0 else 0

        # Optional: Sharpe bonus
        if self.use_sharpe and len(self.returns_history) >= 5:
            mean_return = np.mean(self.returns_history)
            volatility = np.std(self.returns_history)
            sharpe = mean_return / (volatility + 1e-8)
            if sharpe > 0:
                reward += sharpe * self.config.sharpe_bonus

        # Optional: Drawdown penalty
        if self.use_drawdown:
            self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
            drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if drawdown > 0.05:
                reward -= drawdown * self.config.max_drawdown_penalty

        self.prev_portfolio_value = portfolio_value
        return reward


def get_reward_function(reward_type: str = "risk_adjusted", **kwargs) -> RewardFunction:
    """
    Factory function to get reward function by type.

    Args:
        reward_type: Type of reward function
        **kwargs: Additional arguments for reward function

    Returns:
        RewardFunction instance
    """
    if reward_type == "simple":
        return SimpleReturnReward(**kwargs)
    elif reward_type == "risk_adjusted":
        return RiskAdjustedReward(**kwargs)
    elif reward_type == "customizable":
        return CustomizableReward(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")



# ==================== CALLBACKS ====================

class TrainingProgressCallback(BaseCallback):

    def __init__(
        self,
        check_freq: int = 1000,
        save_freq: int = 10000,
        save_path: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
        verbose: int = 0
    ):
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



# ==================== TRAINER ====================

if TYPE_CHECKING:
    from .environments import SingleStockTradingEnv

from .agents import create_agent, BaseRLAgent


@dataclass
class TrainingConfig:
    # Environment settings
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = 10000.0

    # Agent settings
    agent_type: str = "ppo"  # ppo, a2c
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # LSTM feature extractor (hybrid architecture)
    use_lstm: bool = False
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    features_dim: int = 128

    # Training settings
    total_timesteps: int = 50000
    reward_type: str = "risk_adjusted"  # simple, risk_adjusted, customizable
    reward_config: Optional[RewardConfig] = None

    # Transaction costs
    transaction_cost_rate: float = 0.001
    slippage_rate: float = 0.0

    # Saving
    save_dir: Optional[str] = None
    save_best: bool = True

    # Early stopping
    reward_threshold: Optional[float] = None

    # Verbosity
    verbose: int = 1


class RLTrainer:
    """
    Orchestrates RL agent training for trading strategies.
    """

    def __init__(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback

        # Initialize components
        self.env = None
        self.agent = None
        self.training_stats = {}

        # Setup save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default save directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(f"data/models/rl/{config.agent_type}_{config.symbol}_{timestamp}")
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def setup_environment(self) -> "SingleStockTradingEnv":
        """
        Create and setup trading environment.

        Returns:
            Configured trading environment
        """
        from .environments import SingleStockTradingEnv

        # Get reward function
        reward_function = get_reward_function(
            self.config.reward_type,
            config=self.config.reward_config
        )

        # Create environment
        env = SingleStockTradingEnv(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_balance=self.config.initial_balance,
            transaction_cost_rate=self.config.transaction_cost_rate,
            slippage_rate=self.config.slippage_rate,
            reward_function=reward_function,
            include_technical_indicators=True
        )

        self.env = env
        return env

    def setup_agent(self) -> BaseRLAgent:
        """
        Create and setup RL agent.

        Returns:
            Configured RL agent
        """
        if self.env is None:
            self.setup_environment()

        # Agent-specific parameters
        agent_params = {
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'verbose': self.config.verbose
        }

        # Add LSTM parameters if enabled
        if self.config.use_lstm:
            agent_params.update({
                'use_lstm': True,
                'lstm_hidden_size': self.config.lstm_hidden_size,
                'lstm_layers': self.config.lstm_layers,
                'features_dim': self.config.features_dim
            })

        # Create agent
        agent = create_agent(
            self.config.agent_type,
            self.env,
            **agent_params
        )

        self.agent = agent
        return agent

    def train(self) -> Dict[str, Any]:
        """
        Train the RL agent.

        Returns:
            Dictionary with training results
        """
        if self.env is None:
            self.setup_environment()

        if self.agent is None:
            self.setup_agent()

        # Create callbacks
        callbacks = create_training_callbacks(
            save_path=self.save_dir,
            progress_callback=self.progress_callback,
            reward_threshold=self.config.reward_threshold,
            verbose=self.config.verbose
        )

        # Start training
        start_time = time.time()

        if self.config.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Starting RL Training")
            print(f"{'='*60}")
            print(f"Stock: {self.config.symbol}")
            agent_name = self.config.agent_type.upper()
            if self.config.use_lstm:
                agent_name += "-LSTM"
            print(f"Agent: {agent_name}")
            print(f"Period: {self.config.start_date} to {self.config.end_date}")
            print(f"Total Timesteps: {self.config.total_timesteps}")
            if self.config.use_lstm:
                print(f"LSTM Features: Enabled (hidden_size={self.config.lstm_hidden_size}, layers={self.config.lstm_layers})")
            print(f"Save Directory: {self.save_dir}")
            print(f"{'='*60}\n")

        # Train the agent
        self.training_stats = self.agent.train(
            env=self.env,
            total_timesteps=self.config.total_timesteps,
            callback=callbacks
        )

        training_time = time.time() - start_time

        # Save final model
        final_model_path = self.save_dir / "final_model.zip"
        self.agent.save(final_model_path)

        # Get callback statistics
        progress_callback = callbacks[0]  # TrainingProgressCallback
        training_stats_detailed = progress_callback.get_training_stats()

        # Compile results
        results = {
            'success': True,
            'training_time': training_time,
            'total_timesteps': self.config.total_timesteps,
            'total_episodes': training_stats_detailed['total_episodes'],
            'final_model_path': str(final_model_path),
            'save_dir': str(self.save_dir),
            'config': self.config.__dict__,
            'training_stats': training_stats_detailed
        }

        if self.config.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Time: {training_time:.1f}s")
            print(f"Episodes: {results['total_episodes']}")
            print(f"Model saved to: {final_model_path}")
            print(f"{'='*60}\n")

        return results

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate trained agent.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy

        Returns:
            Evaluation results
        """
        if self.agent is None or not self.agent.is_trained:
            raise ValueError("Agent must be trained before evaluation")

        if self.env is None:
            self.setup_environment()

        episode_rewards = []
        episode_lengths = []
        portfolio_values = []

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_values.append(self.env.portfolio_value)

        import numpy as np

        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_portfolio_value': np.mean(portfolio_values),
            'std_portfolio_value': np.std(portfolio_values),
            'episode_rewards': episode_rewards,
            'portfolio_values': portfolio_values
        }

        return results

    def save_config(self):
        """Save training configuration to JSON."""
        import json

        config_path = self.save_dir / "training_config.json"
        config_dict = self.config.__dict__.copy()

        # Convert non-serializable objects
        if 'reward_config' in config_dict and config_dict['reward_config'] is not None:
            config_dict['reward_config'] = config_dict['reward_config'].__dict__

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @staticmethod
    def load_agent(
        model_path: Path,
        agent_type: str,
        env: Optional["SingleStockTradingEnv"] = None
    ):
        """
        Load a trained agent from disk.

        Args:
            model_path: Path to saved model
            agent_type: Type of agent (e.g., 'ppo', 'a2c')
            env: Optional environment (not used, kept for compatibility)

        Returns:
            Loaded agent (stable-baselines3 model)
        """
        from stable_baselines3 import PPO, A2C

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load using stable-baselines3 directly
        if agent_type.lower() == 'ppo':
            agent = PPO.load(str(model_path))
        elif agent_type.lower() == 'a2c':
            agent = A2C.load(str(model_path))
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        # Add is_trained attribute for compatibility with backtest code
        agent.is_trained = True

        return agent
