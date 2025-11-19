"""
Shared Environment Factory

Provides unified environment configuration and creation for training, backtesting,
and live trading to ensure consistency across all modes.
"""

from dataclasses import dataclass, field
from typing import Optional
from .environments import EnhancedTradingEnv
from .improvements import EnhancedRewardConfig, CurriculumManager


@dataclass
class EnvConfig:
    """
    Unified environment configuration for training, backtesting, and live trading.

    This ensures consistent environment setup across all modes and prevents
    train-test mismatch issues.
    """
    # Required parameters
    symbol: str
    start_date: str
    end_date: str

    # Portfolio parameters
    initial_balance: float = 100000.0
    max_position_size: int = 1000  # Maximum shares
    max_position_pct: float = 80.0  # Max position as % of portfolio

    # Cost parameters
    transaction_cost_rate: float = 0.0005  # 0.05% per trade (DQN-optimized)
    slippage_rate: float = 0.0005  # 0.05% slippage

    # Observation parameters
    lookback_window: int = 60
    include_technical_indicators: bool = True
    include_trend_indicators: bool = False  # Trend indicators for LSTM PPO (SMA_Trend, EMA_Crossover, Price_Momentum)

    # Enhancement flags
    use_action_masking: bool = True
    use_enhanced_rewards: bool = True
    use_adaptive_sizing: bool = True
    use_improved_actions: bool = True

    # Optional components
    reward_config: Optional[EnhancedRewardConfig] = None
    curriculum_manager: Optional[CurriculumManager] = None
    enable_diagnostics: bool = True

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_balance': self.initial_balance,
            'max_position_size': self.max_position_size,
            'max_position_pct': self.max_position_pct,
            'transaction_cost_rate': self.transaction_cost_rate,
            'slippage_rate': self.slippage_rate,
            'lookback_window': self.lookback_window,
            'include_technical_indicators': self.include_technical_indicators,
            'include_trend_indicators': self.include_trend_indicators,
            'use_action_masking': self.use_action_masking,
            'use_enhanced_rewards': self.use_enhanced_rewards,
            'use_adaptive_sizing': self.use_adaptive_sizing,
            'use_improved_actions': self.use_improved_actions,
            'enable_diagnostics': self.enable_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvConfig":
        """Create from dictionary (exclude reward_config and curriculum_manager)."""
        # Filter to only include fields that exist in EnvConfig
        valid_fields = {
            'symbol', 'start_date', 'end_date', 'initial_balance',
            'max_position_size', 'max_position_pct', 'transaction_cost_rate',
            'slippage_rate', 'lookback_window', 'include_technical_indicators',
            'include_trend_indicators', 'use_action_masking', 'use_enhanced_rewards',
            'use_adaptive_sizing', 'use_improved_actions', 'enable_diagnostics'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def create_enhanced_env(config: EnvConfig) -> EnhancedTradingEnv:
    """
    Create enhanced trading environment with consistent configuration.

    This factory function ensures that training, backtesting, and live trading
    use the same environment setup, preventing subtle bugs from configuration
    differences.

    Args:
        config: EnvConfig with all environment parameters

    Returns:
        Configured EnhancedTradingEnv instance
    """
    # Use provided reward config or create default
    reward_config = config.reward_config if config.reward_config else EnhancedRewardConfig()

    return EnhancedTradingEnv(
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_balance=config.initial_balance,
        transaction_cost_rate=config.transaction_cost_rate,
        slippage_rate=config.slippage_rate,
        max_position_size=config.max_position_size,
        max_position_pct=config.max_position_pct,
        lookback_window=config.lookback_window,
        include_technical_indicators=config.include_technical_indicators,
        include_trend_indicators=config.include_trend_indicators,
        use_action_masking=config.use_action_masking,
        use_enhanced_rewards=config.use_enhanced_rewards,
        use_adaptive_sizing=config.use_adaptive_sizing,
        use_improved_actions=config.use_improved_actions,
        reward_config=reward_config,
        curriculum_manager=config.curriculum_manager,
        enable_diagnostics=config.enable_diagnostics
    )
