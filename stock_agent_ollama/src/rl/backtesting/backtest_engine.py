"""
Backtesting engine for evaluating trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from ..agents import BaseRLAgent
from ..environments import SingleStockTradingEnv
from .metrics_calculator import MetricsCalculator, PerformanceMetrics


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    transaction_cost_rate: float = 0.001
    slippage_rate: float = 0.0
    risk_free_rate: float = 0.0


@dataclass
class BacktestResult:
    """Results from backtesting."""
    config: BacktestConfig
    metrics: PerformanceMetrics
    portfolio_values: np.ndarray
    actions: List[int]
    trades: List[Dict]
    dates: List[datetime]
    equity_curve: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.__dict__,
            'metrics': self.metrics.to_dict(),
            'portfolio_values': self.portfolio_values.tolist(),
            'actions': self.actions,
            'trades': self.trades,
            'dates': [d.isoformat() for d in self.dates],
        }


class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.env = None

    def setup_environment(self) -> SingleStockTradingEnv:
        """Setup trading environment for backtesting."""
        env = SingleStockTradingEnv(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_balance=self.config.initial_balance,
            transaction_cost_rate=self.config.transaction_cost_rate,
            slippage_rate=self.config.slippage_rate,
            include_technical_indicators=True
        )
        self.env = env
        return env

    def run_agent_backtest(
        self,
        agent: BaseRLAgent,
        deterministic: bool = True
    ) -> BacktestResult:
        """
        Run backtest for RL agent.

        Args:
            agent: Trained RL agent
            deterministic: Whether to use deterministic policy

        Returns:
            BacktestResult with performance metrics
        """
        if not agent.is_trained:
            raise ValueError("Agent must be trained before backtesting")

        if self.env is None:
            self.setup_environment()

        # Run episode
        obs, _ = self.env.reset()
        done = False

        actions = []
        portfolio_values = [self.env.initial_balance]
        trades = []
        dates = []

        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(action)

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Record state
            portfolio_values.append(info['portfolio_value'])
            dates.append(info['date'])

            # Record trades
            if 'trade_info' in info and info['trade_info']['shares_traded'] != 0:
                trades.append({
                    'date': info['date'],
                    'action': 'BUY' if info['trade_info']['shares_traded'] > 0 else 'SELL',
                    'shares': abs(info['trade_info']['shares_traded']),
                    'price': info['price'],
                    'cost': info['trade_info']['total_cost']
                })

        # Calculate metrics
        portfolio_values_array = np.array(portfolio_values)
        metrics = MetricsCalculator.calculate_all_metrics(
            portfolio_values=portfolio_values_array,
            trades=trades,
            initial_balance=self.config.initial_balance,
            risk_free_rate=self.config.risk_free_rate
        )

        # Create equity curve DataFrame
        equity_curve = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_values[1:],  # Skip initial value
            'Action': actions
        })
        equity_curve.set_index('Date', inplace=True)

        # Create result
        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            portfolio_values=portfolio_values_array,
            actions=actions,
            trades=trades,
            dates=dates,
            equity_curve=equity_curve
        )

        return result

    def run_strategy_backtest(
        self,
        strategy_func: Callable,
        **strategy_params
    ) -> BacktestResult:
        """
        Run backtest for custom strategy function.

        Args:
            strategy_func: Function that takes (observation, **params) and returns action
            **strategy_params: Parameters to pass to strategy function

        Returns:
            BacktestResult with performance metrics
        """
        if self.env is None:
            self.setup_environment()

        # Run episode
        obs, _ = self.env.reset()
        done = False

        actions = []
        portfolio_values = [self.env.initial_balance]
        trades = []
        dates = []

        while not done:
            # Get action from strategy
            action = strategy_func(obs, **strategy_params)
            actions.append(action)

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Record state
            portfolio_values.append(info['portfolio_value'])
            dates.append(info['date'])

            # Record trades
            if 'trade_info' in info and info['trade_info']['shares_traded'] != 0:
                trades.append({
                    'date': info['date'],
                    'action': 'BUY' if info['trade_info']['shares_traded'] > 0 else 'SELL',
                    'shares': abs(info['trade_info']['shares_traded']),
                    'price': info['price'],
                    'cost': info['trade_info']['total_cost']
                })

        # Calculate metrics
        portfolio_values_array = np.array(portfolio_values)
        metrics = MetricsCalculator.calculate_all_metrics(
            portfolio_values=portfolio_values_array,
            trades=trades,
            initial_balance=self.config.initial_balance,
            risk_free_rate=self.config.risk_free_rate
        )

        # Create equity curve
        equity_curve = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_values[1:],
            'Action': actions
        })
        equity_curve.set_index('Date', inplace=True)

        # Create result
        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            portfolio_values=portfolio_values_array,
            actions=actions,
            trades=trades,
            dates=dates,
            equity_curve=equity_curve
        )

        return result

    def compare_strategies(
        self,
        strategies: Dict[str, Any],
        deterministic: bool = True
    ) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies.

        Args:
            strategies: Dict mapping strategy name to agent or strategy function
            deterministic: Whether to use deterministic policies for agents

        Returns:
            Dictionary mapping strategy names to backtest results
        """
        results = {}

        for name, strategy in strategies.items():
            if isinstance(strategy, BaseRLAgent):
                result = self.run_agent_backtest(strategy, deterministic)
            elif callable(strategy):
                result = self.run_strategy_backtest(strategy)
            else:
                raise ValueError(f"Unknown strategy type for {name}")

            results[name] = result

        return results

    @staticmethod
    def print_metrics_comparison(
        results: Dict[str, BacktestResult],
        metrics_to_show: Optional[List[str]] = None
    ):
        """
        Print comparison table of metrics.

        Args:
            results: Dictionary of backtest results
            metrics_to_show: List of metrics to display (None = all)
        """
        if metrics_to_show is None:
            metrics_to_show = [
                'cumulative_return',
                'annualized_return',
                'sharpe_ratio',
                'sortino_ratio',
                'max_drawdown',
                'volatility',
                'win_rate',
                'total_trades'
            ]

        # Create comparison DataFrame
        comparison_data = {}
        for name, result in results.items():
            metrics_dict = result.metrics.to_dict()
            comparison_data[name] = {k: metrics_dict[k] for k in metrics_to_show if k in metrics_dict}

        df = pd.DataFrame(comparison_data).T

        # Format for display
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        print(df.to_string())
        print("="*80 + "\n")

        return df
