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
        Shows portfolio value on left axis and return percentage on right axis.

        Args:
            results: Dictionary mapping strategy names to results
            title: Plot title

        Returns:
            Plotly figure
        """
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = ['blue', 'purple', 'orange', 'brown', 'darkgray', 'navy']

        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]

            # Normalize to percentage return
            initial_value = result.portfolio_values[0]
            normalized_values = ((result.portfolio_values / initial_value) - 1) * 100

            # Plot portfolio value on left axis
            fig.add_trace(go.Scatter(
                x=result.dates,
                y=result.portfolio_values[1:],
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
                showlegend=True,
                customdata=normalized_values[1:].reshape(-1, 1),
                hovertemplate='<b>%{fullData.name}</b><br>Portfolio: $%{y:,.2f}<br>Return: %{customdata[0]:.2f}%<extra></extra>'
            ), secondary_y=False)

        # Set y-axes titles
        fig.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False)
        fig.update_yaxes(title_text="Return (%)", secondary_y=True)

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            template="plotly_white",
            hovermode='x unified',
            height=500,
            margin=dict(r=150),  # Add right margin for legend
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02  # Position legend outside plot area
            )
        )

        # Add return % traces on right axis (invisible, just for scaling)
        for idx, (name, result) in enumerate(results.items()):
            initial_value = result.portfolio_values[0]
            normalized_values = ((result.portfolio_values / initial_value) - 1) * 100

            fig.add_trace(go.Scatter(
                x=result.dates,
                y=normalized_values[1:],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=0),  # Invisible
                showlegend=False,
                hoverinfo='skip'
            ), secondary_y=True)

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
        colors = ['blue', 'purple', 'orange', 'brown', 'darkgray', 'navy']

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
            0: 'HOLD',
            1: 'BUY_SMALL',
            2: 'BUY_MEDIUM',
            3: 'BUY_LARGE',
            4: 'SELL_PARTIAL',
            5: 'SELL_ALL'
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
    def plot_action_timeline(
        result: BacktestResult,
        title: str = "Action Timeline"
    ) -> go.Figure:
        """
        Plot equity curve with action markers over time.

        Args:
            result: Backtest result
            title: Plot title

        Returns:
            Plotly figure
        """
        action_names = {
            0: 'HOLD',
            1: 'BUY_SMALL',
            2: 'BUY_MEDIUM',
            3: 'BUY_LARGE',
            4: 'SELL_PARTIAL',
            5: 'SELL_ALL'
        }

        action_colors = {
            0: '#9ca3af',    # HOLD - Gray
            1: '#10b981',    # BUY_SMALL - Green
            2: '#059669',    # BUY_MEDIUM - Darker green
            3: '#047857',    # BUY_LARGE - Darkest green
            4: '#f59e0b',    # SELL_PARTIAL - Orange
            5: '#ef4444'     # SELL_ALL - Red
        }

        fig = go.Figure()

        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=result.portfolio_values[1:],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2),
            hovertemplate='Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
        ))

        # Add action markers grouped by type
        for action_id, action_name in action_names.items():
            # Find indices where this action was taken
            action_indices = [i for i, a in enumerate(result.actions) if a == action_id]

            if action_indices:
                action_dates = [result.dates[i] for i in action_indices]
                action_values = [result.portfolio_values[i + 1] for i in action_indices]

                fig.add_trace(go.Scatter(
                    x=action_dates,
                    y=action_values,
                    mode='markers',
                    name=action_name,
                    marker=dict(
                        size=8,
                        color=action_colors[action_id],
                        symbol='circle',
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'{action_name}<br>Date: %{{x}}<br>Value: $%{{y:.2f}}<extra></extra>'
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            hovermode='closest',
            height=450,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    @staticmethod
    def plot_action_comparison(
        results: Dict[str, BacktestResult],
        title: str = "Action Distribution Comparison"
    ) -> go.Figure:
        """
        Plot stacked bar chart comparing action distributions across strategies.

        Args:
            results: Dictionary mapping strategy names to results
            title: Plot title

        Returns:
            Plotly figure
        """
        # Support both 4-action and 6-action spaces
        action_names = {
            0: 'HOLD',
            1: 'BUY_SMALL',
            2: 'BUY_MEDIUM',
            3: 'BUY_LARGE',
            4: 'SELL_PARTIAL',
            5: 'SELL_ALL'
        }

        action_colors = {
            'HOLD': '#9ca3af',       # Gray
            'BUY_SMALL': '#10b981',  # Green
            'BUY_MEDIUM': '#059669', # Darker green
            'BUY_LARGE': '#047857',  # Darkest green
            'SELL_PARTIAL': '#f59e0b', # Orange
            'SELL_ALL': '#ef4444'    # Red
        }

        strategy_names = list(results.keys())

        # Count actions for each strategy
        action_data = {action_name: [] for action_name in action_names.values()}

        for strategy_name in strategy_names:
            result = results[strategy_name]
            action_counts = {action_name: 0 for action_name in action_names.values()}

            for action in result.actions:
                # Convert numpy array to scalar if needed
                if isinstance(action, (np.ndarray, np.generic)):
                    action_scalar = int(action.item())
                else:
                    action_scalar = int(action)
                action_name = action_names.get(action_scalar, f'Action {action_scalar}')
                action_counts[action_name] += 1

            # Convert to percentages
            total_actions = sum(action_counts.values())
            for action_name in action_names.values():
                percentage = (action_counts[action_name] / total_actions * 100) if total_actions > 0 else 0
                action_data[action_name].append(percentage)

        fig = go.Figure()

        # Add bars for each action type (in logical order)
        for action_name in ['HOLD', 'BUY_SMALL', 'BUY_MEDIUM', 'BUY_LARGE', 'SELL_PARTIAL', 'SELL_ALL']:
            if action_name in action_data and any(v > 0 for v in action_data[action_name]):
                fig.add_trace(go.Bar(
                    name=action_name,
                    x=strategy_names,
                    y=action_data[action_name],
                    marker_color=action_colors[action_name],
                    text=[f'{v:.1f}%' for v in action_data[action_name]],
                    textposition='inside',
                    hovertemplate=f'{action_name}: %{{y:.1f}}%<extra></extra>'
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Strategy",
            yaxis_title="Percentage of Actions (%)",
            barmode='stack',
            template="plotly_white",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    @staticmethod
    def plot_price_with_trades(
        results: Dict[str, BacktestResult],
        symbol: str,
        title: str = "Stock Price with Trade Signals"
    ) -> go.Figure:
        """
        Plot stock price with buy/sell markers for multiple strategies.

        Args:
            results: Dictionary mapping strategy names to results
            symbol: Stock symbol
            title: Plot title

        Returns:
            Plotly figure
        """
        from src.tools.stock_fetcher import StockFetcher

        # Get date range from first result
        first_result = list(results.values())[0]
        start_date = first_result.dates[0] if first_result.dates else None
        end_date = first_result.dates[-1] if first_result.dates else None

        if not start_date or not end_date:
            # Return empty figure if no dates
            fig = go.Figure()
            fig.add_annotation(
                text="No date data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Fetch stock price data
        try:
            fetcher = StockFetcher()
            # Convert dates to string format if they're datetime objects
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            stock_data = fetcher.fetch_stock_data(symbol, start_date=start_str, end_date=end_str)

            if stock_data is None or stock_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No price data available for {symbol}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error fetching price data: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = go.Figure()

        # Normalize stock_data index to remove timezone for comparison
        stock_data_tz_naive = stock_data.copy()
        if stock_data_tz_naive.index.tz is not None:
            stock_data_tz_naive.index = stock_data_tz_naive.index.tz_localize(None)

        # Plot stock price (candlestick or line)
        fig.add_trace(go.Scatter(
            x=stock_data_tz_naive.index,
            y=stock_data_tz_naive['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='#94a3b8', width=2),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Action names and colors
        action_names = {
            0: 'HOLD',
            1: 'BUY_SMALL',
            2: 'BUY_MEDIUM',
            3: 'BUY_LARGE',
            4: 'SELL_PARTIAL',
            5: 'SELL_ALL'
        }

        # Use same colors as performance comparison chart
        performance_colors = ['blue', 'purple', 'orange', 'brown', 'darkgray', 'navy']

        # Plot buy/sell markers for each strategy (exclude baselines)
        baseline_strategies = ['Buy & Hold', 'Momentum']

        for idx, (strategy_name, result) in enumerate(results.items()):
            # Skip baseline strategies
            if strategy_name in baseline_strategies:
                continue

            # Use same color as performance comparison
            color = performance_colors[idx % len(performance_colors)]

            # Get buy and sell actions separately
            buy_actions = [1, 2, 3]  # BUY_SMALL, BUY_MEDIUM, BUY_LARGE
            sell_actions = [4, 5]     # SELL_PARTIAL, SELL_ALL

            # Find buy indices
            buy_indices = []
            for i, action in enumerate(result.actions):
                action_scalar = int(action.item()) if isinstance(action, (np.ndarray, np.generic)) else int(action)
                if action_scalar in buy_actions:
                    buy_indices.append(i)

            if buy_indices:
                buy_dates = [result.dates[i] for i in buy_indices]
                # Get corresponding prices from stock_data
                buy_prices = []
                for date in buy_dates:
                    # Ensure date is timezone-naive for comparison
                    date_naive = date.tz_localize(None) if hasattr(date, 'tz') and date.tz is not None else date

                    if date_naive in stock_data_tz_naive.index:
                        buy_prices.append(stock_data_tz_naive.loc[date_naive, 'Close'])
                    else:
                        # Find nearest date
                        nearest_idx = stock_data_tz_naive.index.get_indexer([date_naive], method='nearest')[0]
                        buy_prices.append(stock_data_tz_naive.iloc[nearest_idx]['Close'])

                fig.add_trace(go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name=f'{strategy_name} BUY',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='triangle-up',
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'{strategy_name} BUY<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))

            # Find sell indices
            sell_indices = []
            for i, action in enumerate(result.actions):
                action_scalar = int(action.item()) if isinstance(action, (np.ndarray, np.generic)) else int(action)
                if action_scalar in sell_actions:
                    sell_indices.append(i)

            if sell_indices:
                sell_dates = [result.dates[i] for i in sell_indices]
                # Get corresponding prices from stock_data
                sell_prices = []
                for date in sell_dates:
                    # Ensure date is timezone-naive for comparison
                    date_naive = date.tz_localize(None) if hasattr(date, 'tz') and date.tz is not None else date

                    if date_naive in stock_data_tz_naive.index:
                        sell_prices.append(stock_data_tz_naive.loc[date_naive, 'Close'])
                    else:
                        # Find nearest date
                        nearest_idx = stock_data_tz_naive.index.get_indexer([date_naive], method='nearest')[0]
                        sell_prices.append(stock_data_tz_naive.iloc[nearest_idx]['Close'])

                fig.add_trace(go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name=f'{strategy_name} SELL',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='triangle-down',
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'{strategy_name} SELL<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=f"{symbol} Price ($)",
            template="plotly_white",
            hovermode='closest',
            height=450,
            margin=dict(r=150),  # Add right margin for legend (same as performance chart)
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02  # Position legend outside plot area (same as performance chart)
            ),
            xaxis=dict(
                rangeslider=dict(visible=False),
                showgrid=True,
                gridcolor='#e5e7eb'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#e5e7eb'
            )
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
        action_names = {
            0: 'HOLD', 1: 'BUY_SMALL', 2: 'BUY_MEDIUM',
            3: 'BUY_LARGE', 4: 'SELL_PARTIAL', 5: 'SELL_ALL'
        }
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
