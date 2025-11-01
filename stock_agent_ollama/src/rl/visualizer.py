"""
Visualization tools for RL trading strategies.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .backtesting import BacktestResult


class RLVisualizer:
    """Visualization tools for RL trading analysis."""

    @staticmethod
    def plot_training_progress(
        training_stats: Dict,
        title: str = "Training Progress"
    ) -> go.Figure:
        """
        Plot training progress metrics.

        Args:
            training_stats: Training statistics dictionary
            title: Plot title

        Returns:
            Plotly figure
        """
        timesteps = training_stats.get('timesteps', [])
        mean_rewards = training_stats.get('mean_rewards', [])
        std_rewards = training_stats.get('std_rewards', [])

        fig = go.Figure()

        # Mean reward line
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=mean_rewards,
            mode='lines',
            name='Mean Reward',
            line=dict(color='blue', width=2)
        ))

        # Confidence interval
        if len(std_rewards) > 0:
            upper_bound = np.array(mean_rewards) + np.array(std_rewards)
            lower_bound = np.array(mean_rewards) - np.array(std_rewards)

            fig.add_trace(go.Scatter(
                x=timesteps + timesteps[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='±1 Std Dev'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Timesteps",
            yaxis_title="Reward",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )

        return fig

    @staticmethod
    def plot_equity_curve(
        result: BacktestResult,
        title: str = "Equity Curve"
    ) -> go.Figure:
        """
        Plot equity curve from backtest result.

        Args:
            result: Backtest result
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=result.portfolio_values[1:],  # Skip initial value
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        ))

        # Initial balance line
        fig.add_hline(
            y=result.config.initial_balance,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Balance",
            annotation_position="right"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )

        return fig

    @staticmethod
    def plot_drawdown(
        result: BacktestResult,
        title: str = "Drawdown"
    ) -> go.Figure:
        """
        Plot drawdown chart.

        Args:
            result: Backtest result
            title: Plot title

        Returns:
            Plotly figure
        """
        # Calculate drawdown
        portfolio_values = result.portfolio_values
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max * 100

        fig = go.Figure()

        # Drawdown area
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=drawdown[1:],  # Skip initial value
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)'
        ))

        # Max drawdown line
        max_dd = result.metrics.max_drawdown * 100
        fig.add_hline(
            y=max_dd,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"Max DD: {max_dd:.2f}%",
            annotation_position="left"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            hovermode='x unified',
            height=300
        )

        return fig

    @staticmethod
    def plot_strategy_comparison(
        results: Dict[str, BacktestResult],
        title: str = "Strategy Comparison"
    ) -> go.Figure:
        """
        Plot comparison of multiple strategies.

        Args:
            results: Dictionary mapping strategy names to results
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]

            # Normalize to percentage return
            initial_value = result.portfolio_values[0]
            normalized_values = ((result.portfolio_values / initial_value) - 1) * 100

            fig.add_trace(go.Scatter(
                x=result.dates,
                y=normalized_values[1:],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_white",
            hovermode='x unified',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    @staticmethod
    def plot_metrics_comparison(
        results: Dict[str, BacktestResult],
        metrics: Optional[List[str]] = None,
        title: str = "Metrics Comparison"
    ) -> go.Figure:
        """
        Plot bar chart comparing metrics across strategies.

        Args:
            results: Dictionary mapping strategy names to results
            metrics: List of metrics to compare (None = use default)
            title: Plot title

        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = [
                'total_return_pct',
                'sharpe_ratio',
                'max_drawdown',
                'win_rate'
            ]

        metric_labels = {
            'total_return_pct': 'Total Return (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown (%)',
            'win_rate': 'Win Rate (%)',
            'annualized_return': 'Annual Return (%)',
            'volatility': 'Volatility'
        }

        # Create subplots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=1,
            cols=n_metrics,
            subplot_titles=[metric_labels.get(m, m) for m in metrics]
        )

        strategy_names = list(results.keys())
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for col_idx, metric in enumerate(metrics, 1):
            values = []
            for name in strategy_names:
                metric_value = getattr(results[name].metrics, metric)

                # Convert some metrics to percentage
                if metric == 'max_drawdown':
                    metric_value = abs(metric_value * 100)
                elif metric == 'win_rate':
                    metric_value = metric_value * 100

                values.append(metric_value)

            fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=values,
                    marker_color=[colors[i % len(colors)] for i in range(len(strategy_names))],
                    showlegend=False,
                    text=[f'{v:.2f}' for v in values],
                    textposition='outside'
                ),
                row=1,
                col=col_idx
            )

        fig.update_layout(
            title_text=title,
            template="plotly_white",
            height=400,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_action_distribution(
        result: BacktestResult,
        title: str = "Action Distribution"
    ) -> go.Figure:
        """
        Plot distribution of actions taken.

        Args:
            result: Backtest result
            title: Plot title

        Returns:
            Plotly figure
        """
        action_names = {
            0: 'SELL',
            1: 'HOLD',
            2: 'BUY_SMALL',
            3: 'BUY_LARGE'
        }

        # Count actions
        action_counts = {}
        for action in result.actions:
            action_name = action_names.get(action, f'Action {action}')
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

        fig = go.Figure(data=[
            go.Pie(
                labels=list(action_counts.keys()),
                values=list(action_counts.values()),
                hole=0.3
            )
        ])

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=350
        )

        return fig

    @staticmethod
    def create_comprehensive_report(
        result: BacktestResult,
        strategy_name: str = "Strategy"
    ) -> go.Figure:
        """
        Create comprehensive report with multiple charts.

        Args:
            result: Backtest result
            strategy_name: Name of strategy

        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                f'{strategy_name} - Equity Curve',
                'Action Distribution',
                'Drawdown',
                'Trade Distribution',
                'Returns Histogram',
                'Performance Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=result.dates,
                y=result.portfolio_values[1:],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ),
            row=1,
            col=1
        )

        # 2. Action Distribution
        action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY_SMALL', 3: 'BUY_LARGE'}
        action_counts = {}
        for action in result.actions:
            name = action_names.get(action, f'Action {action}')
            action_counts[name] = action_counts.get(name, 0) + 1

        fig.add_trace(
            go.Pie(
                labels=list(action_counts.keys()),
                values=list(action_counts.values()),
                hole=0.3
            ),
            row=1,
            col=2
        )

        # 3. Drawdown
        portfolio_values = result.portfolio_values
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(
                x=result.dates,
                y=drawdown[1:],
                mode='lines',
                name='Drawdown',
                line=dict(color='red'),
                fill='tozeroy'
            ),
            row=2,
            col=1
        )

        # 4. Trade count per action type
        fig.add_trace(
            go.Bar(
                x=list(action_counts.keys()),
                y=list(action_counts.values()),
                marker_color='lightblue'
            ),
            row=2,
            col=2
        )

        # 5. Returns Histogram
        returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                marker_color='purple',
                opacity=0.7
            ),
            row=3,
            col=1
        )

        # 6. Metrics Table
        metrics = result.metrics
        metrics_table = [
            ['Metric', 'Value'],
            ['Total Return', f'{metrics.total_return_pct:.2f}%'],
            ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
            ['Max Drawdown', f'{abs(metrics.max_drawdown)*100:.2f}%'],
            ['Win Rate', f'{metrics.win_rate*100:.2f}%'],
            ['Total Trades', f'{metrics.total_trades}']
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=metrics_table[0], fill_color='paleturquoise'),
                cells=dict(values=list(zip(*metrics_table[1:])), fill_color='lavender')
            ),
            row=3,
            col=2
        )

        fig.update_layout(
            height=1000,
            showlegend=False,
            title_text=f"Comprehensive Report: {strategy_name}",
            template="plotly_white"
        )

        return fig
