"""
Backtesting engine for evaluating trading strategies.

This module contains the backtest engine and metrics calculator.
"""
import logging
from .environments import EnhancedTradingEnv
from .improvements import EnhancedRewardConfig
from .env_factory import EnvConfig
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reference defaults from EnvConfig (single source of truth)
_ENV_DEFAULTS = {f.name: f.default for f in EnvConfig.__dataclass_fields__.values()}



# ==================== METRICS CALCULATOR ====================

@dataclass
class PerformanceMetrics:
    # Returns
    cumulative_return: float
    annualized_return: float
    total_return_pct: float

    # Risk metrics
    volatility: float
    annualized_volatility: float
    downside_volatility: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float

    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float

    # Additional metrics
    total_fees: float
    final_portfolio_value: float
    initial_portfolio_value: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'cumulative_return': self.cumulative_return,
            'annualized_return': self.annualized_return,
            'total_return_pct': self.total_return_pct,
            'volatility': self.volatility,
            'annualized_volatility': self.annualized_volatility,
            'downside_volatility': self.downside_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_drawdown': self.avg_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_trade': self.avg_trade,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'total_fees': self.total_fees,
            'final_portfolio_value': self.final_portfolio_value,
            'initial_portfolio_value': self.initial_portfolio_value,
        }


class MetricsCalculator:
    """
    Calculate comprehensive performance and risk metrics.
    """

    @staticmethod
    def calculate_returns_metrics(
        portfolio_values: np.ndarray,
        trading_days_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate return-based metrics.

        Args:
            portfolio_values: Array of portfolio values over time
            trading_days_per_year: Number of trading days per year

        Returns:
            Dictionary of return metrics
        """
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Cumulative return
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        # Annualized return
        n_periods = len(portfolio_values) - 1
        n_years = n_periods / trading_days_per_year
        annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Volatility
        volatility = np.std(returns)
        annualized_volatility = volatility * np.sqrt(trading_days_per_year)

        # Downside volatility (for Sortino)
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(trading_days_per_year) if len(downside_returns) > 0 else 0.0

        return {
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'downside_volatility': downside_volatility,
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
        }

    @staticmethod
    def calculate_risk_adjusted_metrics(
        portfolio_values: np.ndarray,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.

        Args:
            portfolio_values: Array of portfolio values
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year

        Returns:
            Dictionary of risk-adjusted metrics
        """
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Annualized metrics
        mean_return = np.mean(returns) * trading_days_per_year
        volatility = np.std(returns) * np.sqrt(trading_days_per_year)

        # Sharpe Ratio
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(trading_days_per_year) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = (mean_return - risk_free_rate) / downside_volatility

        # Calmar Ratio (return / max drawdown)
        max_drawdown = MetricsCalculator.calculate_max_drawdown(portfolio_values)['max_drawdown']
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }

    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            portfolio_values: Array of portfolio values

        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)

        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max

        # Maximum drawdown
        max_drawdown = np.min(drawdown)

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]

        # Calculate max drawdown duration
        # Handle case where drawdown periods may be incomplete (unequal array lengths)
        max_drawdown_duration = 0
        if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
            # If series started in drawdown, skip first end
            if drawdown_starts[0] > drawdown_ends[0]:
                drawdown_ends = drawdown_ends[1:]

            # Only calculate durations for complete drawdown cycles
            min_len = min(len(drawdown_starts), len(drawdown_ends))

            if min_len > 0:
                durations = drawdown_ends[:min_len] - drawdown_starts[:min_len]
                max_drawdown_duration = int(np.max(durations))
            else:
                max_drawdown_duration = 0
        else:
            max_drawdown_duration = 0

        # Average drawdown
        avg_drawdown = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0.0

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': int(max_drawdown_duration),
            'avg_drawdown': avg_drawdown,
            'drawdown_series': drawdown
        }

    @staticmethod
    def calculate_trading_metrics(trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate trading-specific metrics.

        Args:
            trades: List of trade dictionaries with 'action', 'shares', 'price', 'cost'

        Returns:
            Dictionary of trading metrics
        """
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'avg_trade': 0.0, 'largest_win': 0.0, 'largest_loss': 0.0, 'total_fees': 0.0
            }

        total_fees = sum(t.get('cost', 0.0) for t in trades)
        
        position = 0
        avg_cost_basis = 0.0
        realized_profits = []

        for trade in trades:
            shares = trade['shares']
            price = trade['price']

            if trade['action'] == 'BUY':
                total_cost = (position * avg_cost_basis) + (shares * price)
                position += shares
                avg_cost_basis = total_cost / position if position > 0 else 0
            
            elif trade['action'] == 'SELL':
                if position > 0:
                    shares_to_sell = min(shares, position)
                    profit = (price - avg_cost_basis) * shares_to_sell
                    realized_profits.append(profit)
                    position -= shares_to_sell
                    if position == 0:
                        avg_cost_basis = 0.0
        
        if not realized_profits:
            return {
                'total_trades': len(trades), 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'avg_trade': 0.0, 'largest_win': 0.0, 'largest_loss': 0.0, 'total_fees': total_fees
            }

        wins = [p for p in realized_profits if p > 0]
        losses = [p for p in realized_profits if p < 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        
        total_sell_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_sell_trades if total_sell_trades > 0 else 0.0
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
        
        avg_win = total_wins / winning_trades if winning_trades > 0 else 0.0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0.0
        avg_trade = sum(realized_profits) / total_sell_trades if total_sell_trades > 0 else 0.0
        
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        return {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_fees': total_fees
        }

    @staticmethod
    def calculate_all_metrics(
        portfolio_values: np.ndarray,
        trades: List[Dict],
        initial_balance: float,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            portfolio_values: Array of portfolio values
            trades: List of trades
            initial_balance: Initial portfolio value
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year

        Returns:
            PerformanceMetrics object
        """
        # Returns metrics
        returns_metrics = MetricsCalculator.calculate_returns_metrics(
            portfolio_values, trading_days_per_year
        )

        # Risk-adjusted metrics
        risk_adjusted = MetricsCalculator.calculate_risk_adjusted_metrics(
            portfolio_values, risk_free_rate, trading_days_per_year
        )

        # Drawdown metrics
        drawdown_metrics = MetricsCalculator.calculate_max_drawdown(portfolio_values)

        # Trading metrics
        trading_metrics = MetricsCalculator.calculate_trading_metrics(trades)

        # Create PerformanceMetrics object
        metrics = PerformanceMetrics(
            cumulative_return=returns_metrics['cumulative_return'],
            annualized_return=returns_metrics['annualized_return'],
            total_return_pct=returns_metrics['cumulative_return'] * 100,
            volatility=returns_metrics['volatility'],
            annualized_volatility=returns_metrics['annualized_volatility'],
            downside_volatility=returns_metrics['downside_volatility'],
            sharpe_ratio=risk_adjusted['sharpe_ratio'],
            sortino_ratio=risk_adjusted['sortino_ratio'],
            calmar_ratio=risk_adjusted['calmar_ratio'],
            max_drawdown=drawdown_metrics['max_drawdown'],
            max_drawdown_duration=drawdown_metrics['max_drawdown_duration'],
            avg_drawdown=drawdown_metrics['avg_drawdown'],
            total_trades=trading_metrics['total_trades'],
            winning_trades=trading_metrics['winning_trades'],
            losing_trades=trading_metrics['losing_trades'],
            win_rate=trading_metrics['win_rate'],
            profit_factor=trading_metrics['profit_factor'],
            avg_win=trading_metrics['avg_win'],
            avg_loss=trading_metrics['avg_loss'],
            avg_trade=trading_metrics['avg_trade'],
            largest_win=trading_metrics['largest_win'],
            largest_loss=trading_metrics['largest_loss'],
            total_fees=trading_metrics['total_fees'],
            final_portfolio_value=float(portfolio_values[-1]),
            initial_portfolio_value=initial_balance
        )

        return metrics



# ==================== BACKTEST ENGINE ====================

@dataclass
class BacktestConfig:
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = _ENV_DEFAULTS['initial_balance']
    transaction_cost_rate: float = _ENV_DEFAULTS['transaction_cost_rate']
    slippage_rate: float = _ENV_DEFAULTS['slippage_rate']
    risk_free_rate: float = 0.0

    # Enhancement flags (from EnvConfig)
    use_action_masking: bool = _ENV_DEFAULTS['use_action_masking']
    use_enhanced_rewards: bool = _ENV_DEFAULTS['use_enhanced_rewards']
    use_adaptive_sizing: bool = _ENV_DEFAULTS['use_adaptive_sizing']
    use_improved_actions: bool = _ENV_DEFAULTS['use_improved_actions']
    include_trend_indicators: bool = False  # For LSTM models
    max_position_pct: float = _ENV_DEFAULTS['max_position_pct']
    reward_config: Optional[EnhancedRewardConfig] = field(default_factory=lambda: EnhancedRewardConfig())


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

    def setup_environment(self) -> EnhancedTradingEnv:
        """Setup enhanced trading environment for backtesting using shared factory."""
        from .env_factory import EnvConfig, create_enhanced_env

        # Build environment config from backtest config
        env_config = EnvConfig(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_balance=self.config.initial_balance,
            transaction_cost_rate=self.config.transaction_cost_rate,
            slippage_rate=self.config.slippage_rate,
            max_position_size=1000,  # Standard default
            max_position_pct=self.config.max_position_pct,
            include_technical_indicators=True,
            include_trend_indicators=self.config.include_trend_indicators,
            use_action_masking=self.config.use_action_masking,
            use_enhanced_rewards=self.config.use_enhanced_rewards,
            use_adaptive_sizing=self.config.use_adaptive_sizing,
            use_improved_actions=self.config.use_improved_actions,
            reward_config=self.config.reward_config,
            curriculum_manager=None,  # No curriculum in backtesting
            enable_diagnostics=False  # Disable diagnostics for performance
        )

        # Create environment using shared factory
        self.env = create_enhanced_env(env_config)
        return self.env

    def run_agent_backtest(
        self,
        agent: Any,  # Stable-Baselines3 agent (PPO, RecurrentPPO, QRDQN)
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
        # Stable-Baselines3 agents don't have is_trained attribute
        # Skip this check - assume agent is trained if loaded successfully

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

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Record action
            actions.append(int(action.item() if isinstance(action, np.ndarray) else action))


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
            # Check if it's an RL agent (has predict method from Stable-Baselines3)
            if hasattr(strategy, 'predict') and hasattr(strategy, 'policy'):
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
