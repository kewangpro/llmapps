"""
Training orchestrator for RL trading agents.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:
    from ..environments import SingleStockTradingEnv

from ..agents import create_agent, BaseRLAgent
from .callbacks import create_training_callbacks
from .reward_functions import get_reward_function, RewardConfig


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
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
        from ..environments import SingleStockTradingEnv

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
    ) -> BaseRLAgent:
        """
        Load a trained agent from disk.

        Args:
            model_path: Path to saved model
            agent_type: Type of agent
            env: Optional environment (for creating agent)

        Returns:
            Loaded agent
        """
        from ..environments import SingleStockTradingEnv

        # Create dummy environment if not provided
        if env is None:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            env = SingleStockTradingEnv(
                symbol="AAPL",
                start_date=start_date,
                end_date=end_date
            )

        # Create agent
        agent = create_agent(agent_type, env)

        # Load weights
        agent.load(Path(model_path))

        return agent
