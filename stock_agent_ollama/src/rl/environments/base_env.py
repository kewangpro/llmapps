"""
Base trading environment for reinforcement learning.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
from enum import IntEnum


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
        initial_balance: float = 10000.0,
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
