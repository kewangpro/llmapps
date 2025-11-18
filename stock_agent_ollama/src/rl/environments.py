"""
Trading environments for reinforcement learning.

This module contains base trading environment classes, single stock trading implementation,
and enhanced trading environment with action masking and improvements.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
from enum import IntEnum
import logging

from ..tools.stock_fetcher import StockFetcher
from ..tools.technical_analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)


class TradingAction(IntEnum):
    """Discrete trading actions."""
    SELL = 0
    HOLD = 1
    BUY_SMALL = 2
    BUY_LARGE = 3


class BaseTradingEnv(gym.Env, ABC):
    """
    Base class for trading environments.

    This environment follows the Gymnasium API and provides
    a foundation for building trading RL environments.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_balance: float = 100000.0,
        transaction_cost_rate: float = 0.001,
        slippage_rate: float = 0.0,
        max_position_size: int = 1000,
        enable_short_selling: bool = False,
    ):
        """
        Initialize base trading environment.

        Args:
            initial_balance: Starting cash balance
            transaction_cost_rate: Transaction cost as fraction of trade value
            slippage_rate: Slippage as fraction of trade value
            max_position_size: Maximum number of shares to hold
            enable_short_selling: Whether to allow short positions
        """
        super().__init__()

        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.enable_short_selling = enable_short_selling

        # Portfolio state
        self.cash = initial_balance
        self.position = 0  # Number of shares held
        self.portfolio_value = initial_balance
        self.peak_portfolio_value = initial_balance

        # Trading history
        self.trades = []
        self.portfolio_values = []
        self.actions_taken = []

        # Environment state
        self.current_step = 0
        self.prev_action = TradingAction.HOLD

        # Define action and observation spaces (to be set by subclasses)
        self.action_space = gym.spaces.Discrete(len(TradingAction))
        self.observation_space = None  # Set by subclass

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation/state."""
        pass

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for the current step."""
        pass

    def _execute_action(self, action: int, current_price: float) -> Dict[str, float]:
        """
        Execute a trading action.

        Args:
            action: Trading action to execute
            current_price: Current stock price

        Returns:
            Dictionary with execution details
        """
        shares_to_trade = 0
        trade_cost = 0.0
        slippage_cost = 0.0

        if action == TradingAction.SELL:
            # Sell all holdings
            if self.position > 0:
                shares_to_trade = -self.position
        elif action == TradingAction.BUY_SMALL:
            # Buy 10% of available cash
            affordable_shares = int((self.cash * 0.1) / current_price)
            shares_to_trade = min(affordable_shares, self.max_position_size - self.position)
        elif action == TradingAction.BUY_LARGE:
            # Buy 30% of available cash
            affordable_shares = int((self.cash * 0.3) / current_price)
            shares_to_trade = min(affordable_shares, self.max_position_size - self.position)
        # HOLD: shares_to_trade = 0

        # Execute trade
        if shares_to_trade != 0:
            trade_value = abs(shares_to_trade) * current_price

            # Calculate costs
            trade_cost = trade_value * self.transaction_cost_rate
            slippage_cost = trade_value * self.slippage_rate

            total_cost = trade_cost + slippage_cost

            if shares_to_trade > 0:  # Buying
                total_required = trade_value + total_cost
                if total_required <= self.cash:
                    self.cash -= total_required
                    self.position += shares_to_trade
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'shares': shares_to_trade,
                        'price': current_price,
                        'cost': total_cost
                    })
                else:
                    shares_to_trade = 0  # Can't afford
            else:  # Selling
                proceeds = trade_value - total_cost
                self.cash += proceeds
                self.position += shares_to_trade  # shares_to_trade is negative
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': abs(shares_to_trade),
                    'price': current_price,
                    'cost': total_cost
                })

        return {
            'shares_traded': shares_to_trade,
            'trade_cost': trade_cost,
            'slippage_cost': slippage_cost,
            'total_cost': trade_cost + slippage_cost
        }

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        return self.cash + (self.position * current_price)

    def _calculate_reward(
        self,
        action: int,
        current_price: float,
        prev_price: float,
        trade_info: Dict[str, float]
    ) -> float:
        """
        Calculate reward for the current step.
        To be overridden by subclasses with custom reward functions.

        Args:
            action: Action taken
            current_price: Current stock price
            prev_price: Previous stock price
            trade_info: Information about trade execution

        Returns:
            Reward value
        """
        # Simple reward: change in portfolio value
        prev_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
        current_portfolio_value = self._calculate_portfolio_value(current_price)

        return (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset portfolio state
        self.cash = self.initial_balance
        self.position = 0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance

        # Reset history
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.actions_taken = []

        # Reset environment state
        self.current_step = 0
        self.prev_action = TradingAction.HOLD

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass

    def render(self):
        """Render the environment (optional)."""
        if len(self.portfolio_values) > 0:
            print(f"Step: {self.current_step}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Position: {self.position} shares")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Total Return: {(self.portfolio_value / self.initial_balance - 1) * 100:.2f}%")

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the episode.

        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_values) < 2:
            return {}

        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Total return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        # Volatility (annualized, assuming daily data)
        volatility = np.std(returns) * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (np.mean(returns) * 252) / (volatility + 1e-8)

        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = sum(1 for t in self.trades if t.get('action') == 'SELL' and
                           portfolio_values[t['step']] > portfolio_values[t['step'] - 1])
        total_trades = len([t for t in self.trades if t.get('action') in ['BUY', 'SELL']])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': portfolio_values[-1]
        }


class SingleStockTradingEnv(BaseTradingEnv):
    """
    Trading environment for a single stock.

    Features:
    - Uses real historical stock data
    - Includes technical indicators in state space
    - Supports multiple reward functions
    - Handles transaction costs and slippage
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_balance: float = 100000.0,
        transaction_cost_rate: float = 0.001,
        slippage_rate: float = 0.0,
        max_position_size: int = 1000,
        reward_function: Optional[Any] = None,
        lookback_window: int = 60,
        include_technical_indicators: bool = True,
    ):
        """
        Initialize single stock trading environment.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            initial_balance: Starting cash balance
            transaction_cost_rate: Transaction cost as fraction
            slippage_rate: Slippage as fraction
            max_position_size: Maximum shares to hold
            reward_function: Custom reward function
            lookback_window: Number of historical steps to include in state
            include_technical_indicators: Include technical indicators in state
        """
        super().__init__(
            initial_balance=initial_balance,
            transaction_cost_rate=transaction_cost_rate,
            slippage_rate=slippage_rate,
            max_position_size=max_position_size,
        )

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_window = lookback_window
        self.include_technical_indicators = include_technical_indicators

        # Reward function - import here to avoid circular dependency
        if reward_function is None:
            from .rewards import get_reward_function
            self.reward_function = get_reward_function("risk_adjusted")
        else:
            self.reward_function = reward_function

        # Load stock data
        self._load_data()

        # Define observation space
        self._define_observation_space()

    def _load_data(self):
        """Load and prepare stock data."""
        fetcher = StockFetcher()

        # Fetch stock data
        self.data = fetcher.fetch_stock_data(
            self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )

        if self.data is None or len(self.data) < self.lookback_window:
            raise ValueError(f"Insufficient data for {self.symbol}")

        # Calculate technical indicators
        if self.include_technical_indicators:
            self._calculate_indicators()

        # Normalize price data for state representation
        self._normalize_data()

        # Set episode length
        self.max_steps = len(self.data) - self.lookback_window - 1

    def _calculate_indicators(self):
        """Calculate technical indicators."""
        # Calculate individual indicators using TechnicalAnalysis methods
        close_prices = self.data['Close']
        high_prices = self.data['High']
        low_prices = self.data['Low']

        # Calculate RSI
        self.data['RSI'] = TechnicalAnalysis.calculate_rsi(close_prices)

        # Calculate MACD
        macd_indicators = TechnicalAnalysis.calculate_macd(close_prices)
        self.data['MACD'] = macd_indicators['macd']
        self.data['MACD_Signal'] = macd_indicators['macd_signal']
        self.data['MACD_Histogram'] = macd_indicators['macd_histogram']

        # Calculate Bollinger Bands
        bb_indicators = TechnicalAnalysis.calculate_bollinger_bands(close_prices)
        self.data['BB_Upper'] = bb_indicators['bb_upper']
        self.data['BB_Middle'] = bb_indicators['bb_middle']
        self.data['BB_Lower'] = bb_indicators['bb_lower']

        # Calculate Stochastic
        stoch_indicators = TechnicalAnalysis.calculate_stochastic(high_prices, low_prices, close_prices)
        self.data['Stochastic'] = stoch_indicators['stoch_k']  # Use %K line

        # Calculate SMAs
        self.data['SMA_20'] = TechnicalAnalysis.calculate_sma(close_prices, 20)
        self.data['SMA_50'] = TechnicalAnalysis.calculate_sma(close_prices, 50)

        # Calculate EMAs
        self.data['EMA_12'] = TechnicalAnalysis.calculate_ema(close_prices, 12)
        self.data['EMA_26'] = TechnicalAnalysis.calculate_ema(close_prices, 26)

        # Fill NaN values (use newer pandas method)
        self.data = self.data.bfill().ffill()

    def _normalize_data(self):
        """Normalize data for neural network input."""
        # Store original close prices
        self.original_close = self.data['Close'].values

        # Normalize prices (percentage change from first price)
        first_price = self.data['Close'].iloc[0]
        self.data['Close_Normalized'] = (self.data['Close'] - first_price) / first_price

        # Normalize volume
        if 'Volume' in self.data.columns:
            max_volume = self.data['Volume'].max()
            self.data['Volume_Normalized'] = self.data['Volume'] / (max_volume + 1e-8)

        # Normalize technical indicators (if present)
        if self.include_technical_indicators:
            # RSI is already 0-100, normalize to 0-1
            if 'RSI' in self.data.columns:
                self.data['RSI_Normalized'] = self.data['RSI'] / 100.0

            # MACD and Signal - normalize by price
            if 'MACD' in self.data.columns:
                self.data['MACD_Normalized'] = self.data['MACD'] / (self.data['Close'] + 1e-8)
            if 'MACD_Signal' in self.data.columns:
                self.data['MACD_Signal_Normalized'] = self.data['MACD_Signal'] / (self.data['Close'] + 1e-8)

            # Bollinger Bands - already relative
            # Stochastic - already 0-100
            if 'Stochastic' in self.data.columns:
                self.data['Stochastic_Normalized'] = self.data['Stochastic'] / 100.0

    def _define_observation_space(self):
        """Define the observation space."""
        # Base features: price info + portfolio state
        num_features = 5  # close, volume, cash_ratio, position_ratio, portfolio_value_change

        # Add technical indicators
        if self.include_technical_indicators:
            num_features += 5  # RSI, MACD, MACD_Signal, Bollinger, Stochastic

        # Observation is a window of historical data
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, num_features),
            dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Get current observation/state."""
        # Get data window
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window

        # Extract features
        features = []

        # Price and volume
        close_norm = self.data['Close_Normalized'].iloc[start_idx:end_idx].values
        volume_norm = self.data['Volume_Normalized'].iloc[start_idx:end_idx].values if 'Volume_Normalized' in self.data.columns else np.zeros(self.lookback_window)

        features.append(close_norm)
        features.append(volume_norm)

        # Portfolio state (repeated for each timestep in window)
        current_price = self.original_close[end_idx - 1]
        portfolio_value = self._calculate_portfolio_value(current_price)

        cash_ratio = np.full(self.lookback_window, self.cash / portfolio_value if portfolio_value > 0 else 1.0)
        position_ratio = np.full(self.lookback_window, (self.position * current_price) / portfolio_value if portfolio_value > 0 else 0.0)

        # Portfolio value change
        prev_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
        value_change = np.full(self.lookback_window, (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0)

        features.extend([cash_ratio, position_ratio, value_change])

        # Technical indicators
        if self.include_technical_indicators:
            if 'RSI_Normalized' in self.data.columns:
                rsi = self.data['RSI_Normalized'].iloc[start_idx:end_idx].values
                features.append(rsi)

            if 'MACD_Normalized' in self.data.columns:
                macd = self.data['MACD_Normalized'].iloc[start_idx:end_idx].values
                features.append(macd)

            if 'MACD_Signal_Normalized' in self.data.columns:
                macd_signal = self.data['MACD_Signal_Normalized'].iloc[start_idx:end_idx].values
                features.append(macd_signal)

            # Bollinger Band position (0-1)
            if 'BB_Upper' in self.data.columns and 'BB_Lower' in self.data.columns:
                bb_upper = self.data['BB_Upper'].iloc[start_idx:end_idx].values
                bb_lower = self.data['BB_Lower'].iloc[start_idx:end_idx].values
                close_prices = self.data['Close'].iloc[start_idx:end_idx].values
                bb_position = (close_prices - bb_lower) / (bb_upper - bb_lower + 1e-8)
                features.append(bb_position)
            else:
                features.append(np.zeros(self.lookback_window))

            if 'Stochastic_Normalized' in self.data.columns:
                stoch = self.data['Stochastic_Normalized'].iloc[start_idx:end_idx].values
                features.append(stoch)
            else:
                features.append(np.zeros(self.lookback_window))

        # Stack features (shape: lookback_window x num_features)
        observation = np.column_stack(features).astype(np.float32)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for the current step."""
        current_idx = self.current_step + self.lookback_window
        current_price = self.original_close[current_idx]

        return {
            'step': self.current_step,
            'date': self.data.index[current_idx],
            'price': current_price,
            'cash': self.cash,
            'position': self.position,
            'portfolio_value': self._calculate_portfolio_value(current_price),
            'total_trades': len(self.trades)
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0: SELL, 1: HOLD, 2: BUY_SMALL, 3: BUY_LARGE)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current and previous prices
        current_idx = self.current_step + self.lookback_window
        current_price = self.original_close[current_idx]
        prev_price = self.original_close[current_idx - 1] if current_idx > 0 else current_price

        # Execute action
        trade_info = self._execute_action(action, current_price)

        # Update portfolio value
        prev_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
        current_portfolio_value = self._calculate_portfolio_value(current_price)
        self.portfolio_values.append(current_portfolio_value)
        self.portfolio_value = current_portfolio_value

        # Update peak value
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_portfolio_value)

        # Calculate reward using reward function
        reward = self.reward_function.calculate(
            portfolio_value=current_portfolio_value,
            action=action,
            prev_action=self.prev_action,
            cash=self.cash,
            position=self.position,
            price=current_price,
            prev_price=prev_price
        )

        # Record action
        self.actions_taken.append(action)
        self.prev_action = action

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get observation and info
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._get_info()
        info['trade_info'] = trade_info

        # Add performance metrics if episode is done
        if terminated:
            info['performance_metrics'] = self.get_performance_metrics()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        observation, info = super().reset(seed=seed, options=options)

        # Reset reward function
        self.reward_function.reset()

        return observation, info


# ==============================================================================
# ENHANCED TRADING ENVIRONMENT
# ==============================================================================

# Import improvements components (after other definitions to avoid circular imports)
from .improvements import (
    ActionMasker,
    EnhancedRewardFunction,
    EnhancedRewardConfig,
    AdaptiveActionSizer,
    CurriculumManager,
    TrainingDiagnostics,
    ImprovedTradingAction
)


class EnhancedTradingEnv(SingleStockTradingEnv):
    """
    Enhanced single stock trading environment with:
    - Action masking to prevent invalid actions
    - Enhanced reward shaping
    - Adaptive action sizing
    - Curriculum learning support
    - Better observation space
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_balance: float = 100000.0,
        transaction_cost_rate: float = 0.001,
        slippage_rate: float = 0.0,
        max_position_size: int = 1000,
        max_position_pct: float = 40.0,
        lookback_window: int = 60,
        include_technical_indicators: bool = True,
        # Enhancement parameters
        use_action_masking: bool = True,
        use_enhanced_rewards: bool = True,
        use_adaptive_sizing: bool = True,
        use_improved_actions: bool = True,
        reward_config: Optional[EnhancedRewardConfig] = None,
        curriculum_manager: Optional[CurriculumManager] = None,
        enable_diagnostics: bool = True,
        **kwargs
    ):
        """
        Initialize enhanced trading environment.

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_balance: Starting cash
            transaction_cost_rate: Transaction cost fraction
            slippage_rate: Slippage fraction
            max_position_size: Max shares to hold
            max_position_pct: Max position as % of portfolio value
            lookback_window: Historical window size
            include_technical_indicators: Include technical indicators
            use_action_masking: Enable action masking
            use_enhanced_rewards: Use enhanced reward function
            use_adaptive_sizing: Use adaptive position sizing
            use_improved_actions: Use improved action space (6 actions)
            reward_config: Custom reward configuration
            curriculum_manager: Curriculum learning manager
            enable_diagnostics: Enable training diagnostics
        """
        # Store enhancement flags
        self.use_action_masking = use_action_masking
        self.use_enhanced_rewards = use_enhanced_rewards
        self.use_adaptive_sizing = use_adaptive_sizing
        self.use_improved_actions = use_improved_actions
        self.max_position_pct = max_position_pct
        self.enable_diagnostics = enable_diagnostics

        # Initialize components BEFORE super().__init__
        self.action_masker = ActionMasker(use_improved_actions) if use_action_masking else None

        if use_enhanced_rewards:
            self.enhanced_reward_fn = EnhancedRewardFunction(
                config=reward_config or EnhancedRewardConfig(),
                action_masker=self.action_masker,
                use_improved_actions=use_improved_actions
            )
        else:
            self.enhanced_reward_fn = None

        self.adaptive_sizer = AdaptiveActionSizer() if use_adaptive_sizing else None
        self.curriculum_manager = curriculum_manager

        # Diagnostics
        n_actions = len(ImprovedTradingAction) if use_improved_actions else len(TradingAction)
        self.diagnostics = TrainingDiagnostics(n_actions) if enable_diagnostics else None

        # Initialize parent class WITHOUT reward_function
        # We'll override _calculate_reward instead
        super().__init__(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            transaction_cost_rate=transaction_cost_rate,
            slippage_rate=slippage_rate,
            max_position_size=max_position_size,
            reward_function=None,  # We'll handle this ourselves
            lookback_window=lookback_window,
            include_technical_indicators=include_technical_indicators
        )

        # Override action space if using improved actions
        if use_improved_actions:
            self.action_space = gym.spaces.Discrete(len(ImprovedTradingAction))

        # Track volatility for adaptive sizing
        self._recent_volatility = 0.02

    def _execute_action(self, action: int, current_price: float) -> Dict[str, float]:
        """
        Execute trading action with enhancements.

        Overrides parent to use adaptive sizing.
        """
        shares_to_trade = 0
        trade_cost = 0.0
        slippage_cost = 0.0

        # Use adaptive sizing if enabled
        if self.use_adaptive_sizing and self.adaptive_sizer:
            if self.use_improved_actions:
                # Improved action space
                if action in [1, 2, 3]:  # BUY actions
                    shares_to_trade = self.adaptive_sizer.get_buy_size(
                        action=action,
                        cash=self.cash,
                        price=current_price,
                        position=self.position,
                        portfolio_value=self._calculate_portfolio_value(current_price),
                        max_position_pct=self.max_position_pct,
                        volatility=self._recent_volatility,
                        use_improved_actions=True
                    )
                elif action in [4, 5]:  # SELL actions
                    shares_to_trade = -self.adaptive_sizer.get_sell_size(
                        action=action,
                        position=self.position,
                        use_improved_actions=True
                    )
                # action == 0 is HOLD, shares_to_trade = 0
            else:
                # Standard action space
                if action == TradingAction.SELL:
                    if self.position > 0:
                        shares_to_trade = -self.position
                elif action == TradingAction.BUY_SMALL:
                    shares_to_trade = self.adaptive_sizer.get_buy_size(
                        action=action,
                        cash=self.cash,
                        price=current_price,
                        position=self.position,
                        portfolio_value=self._calculate_portfolio_value(current_price),
                        max_position_pct=self.max_position_pct,
                        volatility=self._recent_volatility,
                        use_improved_actions=False
                    )
                elif action == TradingAction.BUY_LARGE:
                    shares_to_trade = self.adaptive_sizer.get_buy_size(
                        action=action,
                        cash=self.cash,
                        price=current_price,
                        position=self.position,
                        portfolio_value=self._calculate_portfolio_value(current_price),
                        max_position_pct=self.max_position_pct,
                        volatility=self._recent_volatility,
                        use_improved_actions=False
                    )
        else:
            # Use parent class logic
            return super()._execute_action(action, current_price)

        # Execute trade (same as parent)
        if shares_to_trade != 0:
            trade_value = abs(shares_to_trade) * current_price
            trade_cost = trade_value * self.transaction_cost_rate
            slippage_cost = trade_value * self.slippage_rate
            total_cost = trade_cost + slippage_cost

            if shares_to_trade > 0:  # Buying
                total_required = trade_value + total_cost
                if total_required <= self.cash:
                    self.cash -= total_required
                    self.position += shares_to_trade
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'shares': shares_to_trade,
                        'price': current_price,
                        'cost': total_cost
                    })
                else:
                    shares_to_trade = 0  # Can't afford
            else:  # Selling
                proceeds = trade_value - total_cost
                self.cash += proceeds
                self.position += shares_to_trade  # negative
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': abs(shares_to_trade),
                    'price': current_price,
                    'cost': total_cost
                })

        return {
            'shares_traded': shares_to_trade,
            'trade_cost': trade_cost,
            'slippage_cost': slippage_cost,
            'total_cost': trade_cost + slippage_cost
        }

    def _calculate_reward(
        self,
        action: int,
        current_price: float,
        prev_price: float,
        trade_info: Dict[str, float]
    ) -> float:
        """
        Calculate reward using enhanced function if enabled.

        Overrides parent method.
        """
        current_portfolio_value = self._calculate_portfolio_value(current_price)

        if self.use_enhanced_rewards and self.enhanced_reward_fn:
            reward = self.enhanced_reward_fn.calculate(
                portfolio_value=current_portfolio_value,
                action=action,
                prev_action=self.prev_action,
                cash=self.cash,
                position=self.position,
                price=current_price,
                prev_price=prev_price,
                max_position_size=self.max_position_size,
                max_position_pct=self.max_position_pct
            )
        else:
            # Use parent's reward function
            if self.reward_function:
                reward = self.reward_function.calculate(
                    portfolio_value=current_portfolio_value,
                    action=action,
                    prev_action=self.prev_action,
                    cash=self.cash,
                    position=self.position,
                    price=current_price,
                    prev_price=prev_price
                )
            else:
                # Simple return-based reward
                prev_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
                reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value

        return reward

    def get_action_mask(self) -> np.ndarray:
        """
        Get current action mask.

        Returns:
            Binary mask of valid actions
        """
        if not self.use_action_masking or not self.action_masker:
            # All actions valid
            n_actions = len(ImprovedTradingAction) if self.use_improved_actions else len(TradingAction)
            return np.ones(n_actions, dtype=np.float32)

        current_idx = self.current_step + self.lookback_window
        current_price = self.original_close[current_idx]
        portfolio_value = self._calculate_portfolio_value(current_price)

        return self.action_masker.get_action_mask(
            cash=self.cash,
            position=self.position,
            current_price=current_price,
            portfolio_value=portfolio_value,
            max_position_pct=self.max_position_pct
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute step with enhancements.

        Adds action masking info to step info.
        """
        # Get action mask for diagnostics
        action_mask = self.get_action_mask()
        was_valid = bool(action_mask[action])

        # Record for diagnostics
        if self.diagnostics:
            self.diagnostics.record_action(action, was_valid)

        # Execute parent step
        observation, reward, terminated, truncated, info = super().step(action)

        # Add mask to info
        info['action_mask'] = action_mask
        info['action_was_valid'] = was_valid

        # Update volatility estimate
        if len(self.portfolio_values) >= 20:
            returns = np.diff(self.portfolio_values[-20:]) / self.portfolio_values[-20:-1]
            self._recent_volatility = np.std(returns)

        # Record episode for diagnostics
        if (terminated or truncated) and self.diagnostics:
            portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
            self.diagnostics.record_episode(sum(self.portfolio_values), portfolio_return)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment with curriculum learning support.
        """
        # Reset parent
        observation, info = super().reset(seed=seed, options=options)

        # Reset enhanced reward function
        if self.enhanced_reward_fn:
            self.enhanced_reward_fn.reset()

        # Apply curriculum learning if enabled
        if self.curriculum_manager:
            current_idx = self.lookback_window
            current_price = self.original_close[current_idx]

            # Get initial state from curriculum
            initial_cash, initial_position = self.curriculum_manager.get_initial_state(
                self.initial_balance,
                current_price
            )

            # Override initial state
            self.cash = initial_cash
            self.position = initial_position
            self.portfolio_value = self.cash + (self.position * current_price)

            # Update observation with new state
            observation = self._get_observation()
            info['curriculum_stage'] = self.curriculum_manager.get_current_stage().name

        # Add action mask to initial info
        info['action_mask'] = self.get_action_mask()

        return observation, info

    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """Get training diagnostics summary."""
        if self.diagnostics:
            return self.diagnostics.get_summary()
        return {}

    def print_diagnostics(self):
        """Print diagnostics summary."""
        if self.diagnostics:
            self.diagnostics.print_summary()
