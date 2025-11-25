"""
Reinforcement Learning Trading System

This module provides a complete RL trading framework including:
- Enhanced trading environments with action masking
- RL agents (PPO, RecurrentPPO, SAC, QRDQN) via Stable-Baselines3
- Enhanced training pipeline with curriculum learning
- Backtesting engine with comprehensive metrics
- Baseline strategies for comparison
- Visualization tools
"""

from .environments import SingleStockTradingEnv, BaseTradingEnv, TradingAction, EnhancedTradingEnv
from .training import EnhancedRLTrainer, EnhancedTrainingConfig
from .improvements import (
    ActionMasker,
    EnhancedRewardFunction,
    EnhancedRewardConfig,
    AdaptiveActionSizer,
    CurriculumManager,
    ImprovedTradingAction
)
from .backtesting import BacktestEngine, BacktestConfig, PerformanceMetrics
from .baselines import BuyHoldStrategy, MomentumStrategy
from .visualizer import RLVisualizer
from .callbacks import TrainingProgressCallback, EarlyStoppingCallback, PerformanceMonitorCallback
from .rewards import RewardFunction, RewardConfig, get_reward_function

__version__ = "2.0.0"

__all__ = [
    # Environments
    'SingleStockTradingEnv',
    'BaseTradingEnv',
    'EnhancedTradingEnv',
    'TradingAction',
    'ImprovedTradingAction',

    # Training
    'EnhancedRLTrainer',
    'EnhancedTrainingConfig',

    # Improvements
    'ActionMasker',
    'EnhancedRewardFunction',
    'EnhancedRewardConfig',
    'AdaptiveActionSizer',
    'CurriculumManager',

    # Backtesting
    'BacktestEngine',
    'BacktestConfig',
    'PerformanceMetrics',

    # Baselines
    'BuyHoldStrategy',
    'MomentumStrategy',

    # Visualization
    'RLVisualizer',

    # Callbacks
    'TrainingProgressCallback',
    'EarlyStoppingCallback',
    'PerformanceMonitorCallback',

    # Rewards
    'RewardFunction',
    'RewardConfig',
    'get_reward_function',
]
