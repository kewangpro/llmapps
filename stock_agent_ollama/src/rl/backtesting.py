"""
Backtesting engine for evaluating trading strategies.

This module contains the backtest engine and metrics calculator.
"""
import logging
import json
from pathlib import Path
from .environments import EnhancedTradingEnv
from .improvements import EnhancedRewardConfig
from .env_factory import EnvConfig
from ..config import Config
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ==================== BACKTEST ENGINE ====================

@dataclass
class BacktestConfig:
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = Config.RL_DEFAULT_INITIAL_BALANCE
    transaction_cost_rate: float = Config.RL_TRANSACTION_COST_RATE
    slippage_rate: float = Config.RL_SLIPPAGE_RATE
    risk_free_rate: float = 0.0

    # Enhancement flags
    use_action_masking: bool = Config.RL_USE_ACTION_MASKING
    use_enhanced_rewards: bool = Config.RL_USE_ENHANCED_REWARDS
    use_adaptive_sizing: bool = Config.RL_USE_ADAPTIVE_SIZING
    use_improved_actions: bool = Config.RL_USE_IMPROVED_ACTIONS
    include_trend_indicators: bool = False  # For LSTM models
    max_position_pct: float = Config.RL_MAX_POSITION_PCT
    reward_config: Optional[EnhancedRewardConfig] = field(default_factory=lambda: EnhancedRewardConfig())


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
    total_executed: int  # Non-HOLD actions (BUY/SELL actions)
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
            'total_executed': self.total_executed,
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
        trading_days_per_year: int = 252,
        actions: Optional[List[int]] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            portfolio_values: Array of portfolio values
            trades: List of trades
            initial_balance: Initial portfolio value
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year
            actions: List of actions taken (optional, for calculating executed actions)

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

        # Calculate total executed actions (non-HOLD actions)
        # Action 0 is HOLD in improved action space
        if actions is not None:
            total_executed = sum(1 for action in actions if action != 0)
        else:
            # Fallback to total_trades if actions not provided
            total_executed = trading_metrics['total_trades']

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
            total_executed=total_executed,
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
    initial_balance: float = Config.RL_DEFAULT_INITIAL_BALANCE
    transaction_cost_rate: float = Config.RL_TRANSACTION_COST_RATE
    slippage_rate: float = Config.RL_SLIPPAGE_RATE
    risk_free_rate: float = 0.0

    # Enhancement flags
    use_action_masking: bool = Config.RL_USE_ACTION_MASKING
    use_enhanced_rewards: bool = Config.RL_USE_ENHANCED_REWARDS
    use_adaptive_sizing: bool = Config.RL_USE_ADAPTIVE_SIZING
    use_improved_actions: bool = Config.RL_USE_IMPROVED_ACTIONS
    # New improvement flags
    use_risk_manager: bool = True
    use_regime_detector: bool = True
    use_mtf_features: bool = True
    use_kelly_sizing: bool = True

    include_trend_indicators: bool = False  # For LSTM models
    max_position_pct: float = Config.RL_MAX_POSITION_PCT
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
    paired_trades: List[Dict] = None  # Round-trip trades for P&L validation
    final_cash_balance: float = None  # Cash at end of backtest
    final_position_value: float = None  # Value of open position at end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching the schema of saved backtest_results.json."""
        # Convert trades to JSON-serializable format
        serializable_trades = []
        for trade in self.trades:
            trade_copy = trade.copy()
            # Convert Timestamp objects to ISO format strings
            for date_field in ['date', 'timestamp']:
                if date_field in trade_copy:
                    ts = trade_copy[date_field]
                    if hasattr(ts, 'isoformat'):
                        trade_copy[date_field] = ts.isoformat()
                    elif isinstance(ts, str):
                        trade_copy[date_field] = ts
            serializable_trades.append(trade_copy)

        # Calculate action distribution
        from collections import Counter
        from .types import ImprovedTradingAction
        action_counts = Counter(self.actions)
        
        action_distribution = {}
        for action_id, count in action_counts.items():
            try:
                # Try to use improved action name if applicable
                # Note: This might be inaccurate if using standard actions, but it's for display
                action_name = ImprovedTradingAction(action_id).name
                action_distribution[action_name] = count
            except ValueError:
                action_distribution[f"ACTION_{action_id}"] = count

        # Base dictionary
        result = {
            'config': self.config.__dict__,
            'metrics': self.metrics.to_dict(), # Keep nested metrics for completeness
            'portfolio_values': self.portfolio_values.tolist(),
            'actions': self.actions,
            'trades': serializable_trades,
            'dates': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in self.dates],
            'action_distribution': action_distribution
        }
        
        # Flatten metrics into top-level for compatibility with validation script
        # This matches the structure in save_to_model_dir
        metrics_dict = self.metrics.to_dict()
        for k, v in metrics_dict.items():
            result[k] = v

        if self.paired_trades is not None:
            result['paired_trades'] = self.paired_trades

        # Add cash and position breakdown for validation
        if self.final_cash_balance is not None:
            result['final_cash_balance'] = self.final_cash_balance
        if self.final_position_value is not None:
            result['final_position_value'] = self.final_position_value

        return result

    def save_to_model_dir(self, model_path: Path, agent_type: str):
        """
        Save backtest results to model directory for auto-select feature.

        Args:
            model_path: Path to model file or directory
            agent_type: Algorithm type (ppo, recurrent_ppo, ensemble)
        """
        try:
            # Determine the directory to save to
            if agent_type == 'ensemble':
                # For ensemble, model_path could be the ensemble subdirectory or parent
                if model_path.name == 'ensemble':
                    save_dir = model_path.parent
                elif (model_path / 'ensemble').exists():
                    save_dir = model_path
                else:
                    save_dir = model_path
            else:
                # For PPO/RecurrentPPO, model_path could be .zip file or directory
                save_dir = model_path.parent if model_path.is_file() else model_path

            # Calculate action distribution for validation
            from collections import Counter
            from .types import ImprovedTradingAction
            action_counts = Counter(self.actions)
            
            action_distribution = {}
            for action_id, count in action_counts.items():
                try:
                    action_name = ImprovedTradingAction(action_id).name
                    action_distribution[action_name] = count
                except ValueError:
                    # Fallback for unknown actions
                    action_distribution[f"ACTION_{action_id}"] = count

            # Convert trades to JSON-serializable format
            serializable_trades = []
            for trade in self.trades:
                trade_copy = trade.copy()
                # Convert Timestamp objects to ISO format strings
                for date_field in ['date', 'timestamp']:
                    if date_field in trade_copy:
                        ts = trade_copy[date_field]
                        if hasattr(ts, 'isoformat'):
                            trade_copy[date_field] = ts.isoformat()
                        elif isinstance(ts, str):
                            trade_copy[date_field] = ts
                serializable_trades.append(trade_copy)

            # Convert paired_trades to JSON-serializable format (if available)
            serializable_paired_trades = None
            if self.paired_trades:
                serializable_paired_trades = self.paired_trades  # Already plain dicts

            # Create backtest results JSON with detailed data for validation
            # Convert dates to strings if they're Timestamp objects
            start_date = self.config.start_date
            if hasattr(start_date, 'isoformat'):
                start_date = start_date.strftime('%Y-%m-%d')
            end_date = self.config.end_date
            if hasattr(end_date, 'isoformat'):
                end_date = end_date.strftime('%Y-%m-%d')

            backtest_data = {
                'agent_type': agent_type,
                'backtest_date': datetime.now().isoformat(),
                'backtest_period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                # Performance metrics
                'total_return_pct': self.metrics.total_return_pct,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'sortino_ratio': self.metrics.sortino_ratio,
                'calmar_ratio': self.metrics.calmar_ratio,
                'max_drawdown': self.metrics.max_drawdown,
                'win_rate': self.metrics.win_rate,
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'avg_win': self.metrics.avg_win,
                'avg_loss': self.metrics.avg_loss,
                'profit_factor': self.metrics.profit_factor,
                # Detailed data for validation
                'initial_portfolio_value': float(self.metrics.initial_portfolio_value),
                'final_portfolio_value': float(self.metrics.final_portfolio_value),
                'portfolio_values': self.portfolio_values.tolist(),
                'action_distribution': action_distribution,
                'trades': serializable_trades,
                'dates': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in self.dates]
            }

            # Add paired_trades if available (for P&L validation)
            if serializable_paired_trades:
                backtest_data['paired_trades'] = serializable_paired_trades

            # Add cash and position breakdown for validation
            if self.final_cash_balance is not None:
                backtest_data['final_cash_balance'] = float(self.final_cash_balance)
            if self.final_position_value is not None:
                backtest_data['final_position_value'] = float(self.final_position_value)

            # Save to model directory
            backtest_file = save_dir / "backtest_results.json"
            with open(backtest_file, 'w') as f:
                json.dump(backtest_data, f, indent=2)

            logger.info(f"Saved backtest results to {backtest_file}")

        except Exception as e:
            logger.warning(f"Failed to save backtest results for {agent_type}: {e}")


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
            # New improvements
            use_risk_manager=self.config.use_risk_manager,
            use_regime_detector=self.config.use_regime_detector,
            use_mtf_features=self.config.use_mtf_features,
            use_kelly_sizing=self.config.use_kelly_sizing,
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
        paired_trades = []  # Round-trip trades for P&L validation
        dates = []

        # Track position for paired trade generation
        position_tracker = {
            'shares': 0,
            'total_cost': 0.0,  # Total cost basis (shares * avg_entry_price)
            'avg_entry_price': 0.0,
            'total_commission_paid': 0.0  # Accumulated buy commissions
        }

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
                shares_traded = info['trade_info']['shares_traded']
                price = info['price']
                commission = info['trade_info']['total_cost']

                # Record individual action trade
                trades.append({
                    'date': info['date'],
                    'action': 'BUY' if shares_traded > 0 else 'SELL',
                    'shares': abs(shares_traded),
                    'price': price,
                    'cost': commission,
                    'commission': commission
                })

                # Update position tracker and generate paired trades
                if shares_traded > 0:  # BUY
                    # Update position (costs NOT included in avg price)
                    new_shares = position_tracker['shares'] + shares_traded
                    position_tracker['total_cost'] += shares_traded * price
                    position_tracker['avg_entry_price'] = position_tracker['total_cost'] / new_shares
                    position_tracker['shares'] = new_shares
                    position_tracker['total_commission_paid'] += commission

                else:  # SELL
                    # Generate paired round-trip trade for validation
                    if position_tracker['shares'] > 0:
                        sell_shares = abs(shares_traded)
                        entry_price = position_tracker['avg_entry_price']
                        exit_price = price

                        # Calculate proportional buy commission for this sell
                        # Total buy commission paid is tracked per position
                        if position_tracker['shares'] > 0:
                            buy_commission_for_this_trade = (sell_shares / position_tracker['shares']) * position_tracker['total_commission_paid']
                        else:
                            buy_commission_for_this_trade = 0.0

                        # Calculate P&L: price difference minus ALL commissions (buy + sell)
                        pnl = (exit_price - entry_price) * sell_shares - commission - buy_commission_for_this_trade

                        paired_trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': sell_shares,
                            'commission': commission + buy_commission_for_this_trade,  # Total commission (buy + sell)
                            'pnl': pnl
                        })

                        # Update position
                        position_tracker['shares'] -= sell_shares
                        # Reduce total cost basis proportionally
                        position_tracker['total_cost'] -= sell_shares * entry_price
                        # Reduce commission proportionally
                        position_tracker['total_commission_paid'] -= buy_commission_for_this_trade

                        # Recalculate average entry price for remaining shares
                        if position_tracker['shares'] > 0:
                            position_tracker['avg_entry_price'] = position_tracker['total_cost'] / position_tracker['shares']
                        else:
                            position_tracker['total_cost'] = 0.0
                            position_tracker['avg_entry_price'] = 0.0
                            position_tracker['total_commission_paid'] = 0.0

        # Extract final cash and position from last step
        # info dict from last step contains final state
        final_cash = info.get('cash', 0.0)
        final_position_shares = info.get('position', 0.0)
        final_price = info.get('price', 0.0)
        final_position_value = final_position_shares * final_price

        # Calculate metrics
        portfolio_values_array = np.array(portfolio_values)
        metrics = MetricsCalculator.calculate_all_metrics(
            portfolio_values=portfolio_values_array,
            trades=trades,
            initial_balance=self.config.initial_balance,
            risk_free_rate=self.config.risk_free_rate,
            actions=actions
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
            equity_curve=equity_curve,
            paired_trades=paired_trades if paired_trades else None,
            final_cash_balance=final_cash,
            final_position_value=final_position_value
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
        paired_trades = []  # Round-trip trades for P&L validation
        dates = []

        # Track position for paired trade generation
        position_tracker = {
            'shares': 0,
            'total_cost': 0.0,  # Total cost basis (shares * avg_entry_price)
            'avg_entry_price': 0.0,
            'total_commission_paid': 0.0  # Accumulated buy commissions
        }

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
                shares_traded = info['trade_info']['shares_traded']
                price = info['price']
                commission = info['trade_info']['total_cost']

                # Record individual action trade
                trades.append({
                    'date': info['date'],
                    'action': 'BUY' if shares_traded > 0 else 'SELL',
                    'shares': abs(shares_traded),
                    'price': price,
                    'cost': commission,
                    'commission': commission
                })

                # Update position tracker and generate paired trades
                if shares_traded > 0:  # BUY
                    # Update position (costs NOT included in avg price)
                    new_shares = position_tracker['shares'] + shares_traded
                    position_tracker['total_cost'] += shares_traded * price
                    position_tracker['avg_entry_price'] = position_tracker['total_cost'] / new_shares
                    position_tracker['shares'] = new_shares
                    position_tracker['total_commission_paid'] += commission

                else:  # SELL
                    # Generate paired round-trip trade for validation
                    if position_tracker['shares'] > 0:
                        sell_shares = abs(shares_traded)
                        entry_price = position_tracker['avg_entry_price']
                        exit_price = price

                        # Calculate proportional buy commission for this sell
                        # Total buy commission paid is tracked per position
                        if position_tracker['shares'] > 0:
                            buy_commission_for_this_trade = (sell_shares / position_tracker['shares']) * position_tracker['total_commission_paid']
                        else:
                            buy_commission_for_this_trade = 0.0

                        # Calculate P&L: price difference minus ALL commissions (buy + sell)
                        pnl = (exit_price - entry_price) * sell_shares - commission - buy_commission_for_this_trade

                        paired_trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': sell_shares,
                            'commission': commission + buy_commission_for_this_trade,  # Total commission (buy + sell)
                            'pnl': pnl
                        })

                        # Update position
                        position_tracker['shares'] -= sell_shares
                        # Reduce total cost basis proportionally
                        position_tracker['total_cost'] -= sell_shares * entry_price
                        # Reduce commission proportionally
                        position_tracker['total_commission_paid'] -= buy_commission_for_this_trade

                        # Recalculate average entry price for remaining shares
                        if position_tracker['shares'] > 0:
                            position_tracker['avg_entry_price'] = position_tracker['total_cost'] / position_tracker['shares']
                        else:
                            position_tracker['total_cost'] = 0.0
                            position_tracker['avg_entry_price'] = 0.0
                            position_tracker['total_commission_paid'] = 0.0

        # Extract final cash and position from last step
        final_cash = info.get('cash', 0.0)
        final_position_shares = info.get('position', 0.0)
        final_price = info.get('price', 0.0)
        final_position_value = final_position_shares * final_price

        # Calculate metrics
        portfolio_values_array = np.array(portfolio_values)
        metrics = MetricsCalculator.calculate_all_metrics(
            portfolio_values=portfolio_values_array,
            trades=trades,
            initial_balance=self.config.initial_balance,
            risk_free_rate=self.config.risk_free_rate,
            actions=actions
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
            equity_curve=equity_curve,
            paired_trades=paired_trades if paired_trades else None,
            final_cash_balance=final_cash,
            final_position_value=final_position_value
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
