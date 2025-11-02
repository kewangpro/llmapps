"""
Reinforcement Learning Trading System

This module provides a complete RL trading framework including:
- Trading environments (Gymnasium-based)
- RL agents (PPO, A2C)
- Training pipeline with callbacks
- Backtesting engine with comprehensive metrics
- Baseline strategies for comparison
- Visualization tools
"""

from .environments import SingleStockTradingEnv, BaseTradingEnv, TradingAction
from .agents import PPOAgent, A2CAgent, create_agent
from .training import RLTrainer, TrainingConfig, RewardConfig, get_reward_function
from .backtesting import BacktestEngine, BacktestConfig, PerformanceMetrics
from .baselines import BuyHoldStrategy, MomentumStrategy
from .visualizer import RLVisualizer

__version__ = "1.0.0"

__all__ = [
    # Environments
    'SingleStockTradingEnv',
    'BaseTradingEnv',
    'TradingAction',

    # Agents
    'PPOAgent',
    'A2CAgent',
    'create_agent',

    # Training
    'RLTrainer',
    'TrainingConfig',
    'RewardConfig',

    # Backtesting
    'BacktestEngine',
    'BacktestConfig',
    'PerformanceMetrics',

    # Baselines
    'BuyHoldStrategy',
    'MomentumStrategy',

    # Visualization
    'RLVisualizer',
]
