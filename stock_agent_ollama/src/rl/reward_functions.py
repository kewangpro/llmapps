"""
Reward function implementations for RL trading agents.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward function calculation."""
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
