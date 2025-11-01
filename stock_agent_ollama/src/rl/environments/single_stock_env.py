"""
Single stock trading environment for reinforcement learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import gymnasium as gym

from .base_env import BaseTradingEnv, TradingAction
from ...tools.stock_fetcher import StockFetcher
from ...tools.technical_analysis import TechnicalAnalysis
from ..training.reward_functions import RewardFunction, get_reward_function


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
        initial_balance: float = 10000.0,
        transaction_cost_rate: float = 0.001,
        slippage_rate: float = 0.0,
        max_position_size: int = 1000,
        reward_function: Optional[RewardFunction] = None,
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

        # Reward function
        if reward_function is None:
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
