"""
RL Training UI - Enhanced training with comprehensive improvements.

This panel provides:
- Enhanced RL training with action masking, curriculum learning, etc.
- Backtesting with support for improved action spaces
"""

import panel as pn
import param
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import threading

from src.ui.design_system import HTMLComponents

logger = logging.getLogger(__name__)


class RLTrainingPanel(param.Parameterized):
    """RL training panel with enhanced training system."""

    def __init__(self, **params):
        super().__init__(**params)
        self.trainer = None
        self.is_training = False
        self._create_ui()

    def _create_ui(self):
        """Create UI components."""
        # Symbol input with autocomplete
        self.symbol_input = pn.widgets.AutocompleteInput(
            name='',
            value='GOOGL',
            options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'ORCL',
                     'TEAM', 'NFLX', 'AMD', 'INTC', 'QCOM', 'CRM', 'ADBE', 'PYPL'],
            placeholder='Enter symbol...',
            case_sensitive=False,
            width=120,
            height=35,
            min_characters=1
        )

        # Agent type
        self.agent_type = pn.widgets.RadioButtonGroup(
            name='Agent',
            options=['PPO', 'A2C'],
            value='PPO',
            button_type='primary',
            button_style='outline'
        )

        # === IMPROVEMENT OPTIONS ===
        # Note: Action Masking and 6-Action Space are always enabled for safety and performance

        # Enhanced rewards
        self.use_enhanced_rewards = pn.widgets.Checkbox(
            name='Enhanced Rewards',
            value=True
        )

        # Adaptive sizing
        self.use_adaptive_sizing = pn.widgets.Checkbox(
            name='Adaptive Sizing',
            value=True
        )

        # Curriculum learning
        self.use_curriculum = pn.widgets.Checkbox(
            name='Curriculum Learning',
            value=True
        )

        # Training parameters
        self.training_days = pn.widgets.IntSlider(
            name='Training Period (days)',
            start=180,
            end=1095,
            value=1095,
            step=30,
            width=250
        )

        self.timesteps = pn.widgets.IntSlider(
            name='Training Steps',
            start=50000,
            end=500000,
            value=300000,
            step=10000,
            width=250
        )

        # Exploration bonus (entropy coefficient)
        self.ent_coef = pn.widgets.FloatSlider(
            name='Exploration Bonus (ent_coef)',
            start=0.0,
            end=0.1,
            value=0.02,
            step=0.01,
            width=250
        )

        # Train button
        self.train_button = pn.widgets.Button(
            name="🚀 Start Training",
            button_type="success",
            width=150,
            height=40
        )
        self.train_button.on_click(self._start_training)

        # Backtest button
        self.backtest_button = pn.widgets.Button(
            name="📊 Run Backtest",
            button_type="primary",
            width=150,
            height=40
        )
        self.backtest_button.on_click(self._run_backtest)

        # Progress
        self.progress_bar = pn.indicators.Progress(
            name='Progress',
            value=0,
            max=100,
            width=300,
            bar_color='success',
            visible=False
        )

        # Model status display
        self.model_status_pane = pn.pane.HTML("", sizing_mode="stretch_width")

        # Results panel
        self.results_panel = pn.Column(sizing_mode="stretch_width", min_height=250)

    def _start_training(self, event):
        """Start RL training."""
        if self.is_training:
            pn.state.notifications.warning("Training already in progress", duration=3000)
            return

        self.is_training = True
        self.train_button.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.results_panel.clear()
        self.model_status_pane.object = ""  # Clear model info when starting new training

        symbol = self.symbol_input.value.strip().upper()
        if not symbol:
            pn.state.notifications.error("Please enter a stock symbol", duration=3000)
            self.is_training = False
            self.train_button.disabled = False
            return

        pn.state.notifications.info(
            f"Starting {self.agent_type.value} training on {symbol}...",
            duration=3000
        )

        def train_thread():
            try:
                from src.rl.training import EnhancedRLTrainer, EnhancedTrainingConfig
                from src.rl.improvements import EnhancedRewardConfig

                # Setup dates
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=self.training_days.value)).strftime("%Y-%m-%d")

                reward_config = EnhancedRewardConfig(
                    invalid_action_penalty=-0.5,
                    profitable_trade_bonus=0.2,
                    use_action_shaping=True,
                    min_hold_steps=5
                )

                config = EnhancedTrainingConfig(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    agent_type=self.agent_type.value.lower(),
                    total_timesteps=self.timesteps.value,
                    ent_coef=self.ent_coef.value,

                    # Improvements (Action Masking and 6-Action Space always enabled)
                    use_action_masking=True,
                    use_enhanced_rewards=self.use_enhanced_rewards.value,
                    use_adaptive_sizing=self.use_adaptive_sizing.value,
                    use_improved_actions=True,
                    use_curriculum_learning=self.use_curriculum.value,
                    enable_diagnostics=True,

                    reward_config=reward_config,
                    verbose=1
                )

                trainer = EnhancedRLTrainer(config, progress_callback=self._update_progress)
                results = trainer.train()

                # Display results
                pn.state.execute(lambda: self._display_training_results(results, symbol))
                pn.state.execute(lambda: pn.state.notifications.success(
                    "Training complete!", duration=5000
                ))

            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                pn.state.execute(lambda: pn.state.notifications.error(
                    f"Training failed: {str(e)}", duration=5000
                ))
                pn.state.execute(lambda: self.results_panel.append(pn.pane.Alert(
                    f"**Error:** {str(e)}",
                    alert_type="danger"
                )))

            finally:
                pn.state.execute(lambda: setattr(self, 'is_training', False))
                pn.state.execute(lambda: setattr(self.train_button, 'disabled', False))
                pn.state.execute(lambda: setattr(self.progress_bar, 'visible', False))

        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()

    def _update_progress(self, progress_data: Dict):
        """Update training progress."""
        if 'timestep' in progress_data:
            progress = int((progress_data['timestep'] / self.timesteps.value) * 100)
            pn.state.execute(lambda: setattr(self.progress_bar, 'value', min(progress, 100)))

    def _display_training_results(self, results: Dict, symbol: str):
        """Display training results."""
        self.results_panel.clear()

        # Determine agent name (always uses 6-action space now)
        agent_name = self.agent_type.value

        # Summary card
        summary_html = f"""
        <div style='background: #F8F9FA;
                    border: 1px solid #DEE2E6;
                    border-left: 4px solid #0F9D58;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
            <h3 style='margin: 0 0 10px 0; color: #212529;'>✅ Training Complete</h3>
            <div style='font-size: 13px; color: #495057; line-height: 1.6;'>
                <strong>Agent:</strong> {agent_name}<br>
                <strong>Stock:</strong> {symbol}<br>
                <strong>Episodes:</strong> {results.get('total_episodes', 'N/A')}<br>
                <strong>Time:</strong> {results.get('training_time', 0):.1f}s<br>
                <strong>Model:</strong> {Path(results.get('final_model_path', '')).name}
            </div>
        """

        # Add diagnostics if available
        if 'diagnostics' in results:
            diag = results['diagnostics']
            summary_html += f"""
            <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #DEE2E6;'>
                <strong style='color: #212529;'>Training Diagnostics:</strong><br>
                <div style='font-size: 12px; color: #495057; margin-top: 5px;'>
                    Invalid Action Rate: <strong style='color: {"#0F9D58" if diag.get("invalid_action_rate", 0) < 0.05 else "#DC3545"};'>
                        {diag.get('invalid_action_rate', 0):.2%}
                    </strong><br>
                    Mean Episode Reward: <strong>{diag.get('mean_episode_reward', 0):.4f}</strong><br>
                    Mean Portfolio Return: <strong>{diag.get('mean_portfolio_return', 0):.2%}</strong>
                </div>
            </div>
            """

        # Add high-priority training metrics
        if 'training_stats' in results:
            stats = results['training_stats']
            summary_html += f"""
            <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #DEE2E6;'>
                <strong style='color: #212529;'>Training Metrics:</strong><br>
                <div style='font-size: 12px; color: #495057; margin-top: 5px;'>
            """

            # Win Rate
            win_rate = stats.get('win_rate', 0)
            win_rate_color = '#0F9D58' if win_rate >= 50 else '#F59E0B' if win_rate >= 30 else '#DC3545'
            summary_html += f"""
                    Win Rate: <strong style='color: {win_rate_color};'>{win_rate:.1f}%</strong><br>
            """

            # Final/Best Episode Reward
            final_reward = stats.get('final_episode_reward', 0)
            best_reward = stats.get('best_episode_reward', 0)
            summary_html += f"""
                    Final Episode Reward: <strong>{final_reward:.2f}</strong><br>
                    Best Episode Reward: <strong>{best_reward:.2f}</strong><br>
            """

            # Explained Variance (if available)
            if results.get('explained_variance') is not None:
                exp_var = results['explained_variance']
                exp_var_color = '#0F9D58' if exp_var >= 0.7 else '#F59E0B' if exp_var >= 0.5 else '#DC3545'
                summary_html += f"""
                    Explained Variance: <strong style='color: {exp_var_color};'>{exp_var:.3f}</strong>
                """

            summary_html += """
                </div>
            </div>
            """

        summary_html += "</div>"
        self.results_panel.append(pn.pane.HTML(summary_html))

        # Training progress chart
        if 'training_stats' in results:
            try:
                from src.rl import RLVisualizer
                fig = RLVisualizer.plot_training_progress(
                    results['training_stats'],
                    title=f"{agent_name} Training Progress - {symbol}"
                )
                self.results_panel.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))
            except Exception as e:
                logger.error(f"Error plotting training progress: {e}", exc_info=True)

        # Action distribution chart
        if 'training_stats' in results and 'action_distribution' in results['training_stats']:
            try:
                import plotly.graph_objects as go

                stats = results['training_stats']
                action_dist = stats['action_distribution']

                # Define action names and colors
                action_names_map = {
                    0: 'HOLD',
                    1: 'BUY_SMALL',
                    2: 'BUY_MEDIUM',
                    3: 'BUY_LARGE',
                    4: 'SELL_PARTIAL',
                    5: 'SELL_ALL'
                }

                action_colors_map = {
                    0: '#9ca3af',    # HOLD - Gray
                    1: '#10b981',    # BUY_SMALL - Green
                    2: '#059669',    # BUY_MEDIUM - Darker green
                    3: '#047857',    # BUY_LARGE - Darkest green
                    4: '#f59e0b',    # SELL_PARTIAL - Orange
                    5: '#ef4444'     # SELL_ALL - Red
                }

                # Prepare data
                actions = sorted(action_dist.keys())
                labels = [action_names_map.get(a, f'Action {a}') for a in actions]
                values = [action_dist[a] for a in actions]
                colors = [action_colors_map.get(a, '#6b7280') for a in actions]

                # Create pie chart
                fig_actions = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors),
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='auto',
                    hovertemplate='<b>%{label}</b><br>%{value:.1f}% of actions<extra></extra>'
                )])

                fig_actions.update_layout(
                    title=f"Action Distribution During Training - {symbol}",
                    template="plotly_white",
                    height=350,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )

                self.results_panel.append(pn.pane.Plotly(fig_actions, sizing_mode="stretch_width", height=350))

            except Exception as e:
                logger.error(f"Error plotting action distribution: {e}", exc_info=True)

    def _find_latest_model(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Find the most recently trained model for a symbol."""
        models_dir = Path("data/models/rl")
        if not models_dir.exists():
            return None

        # Find all model directories for this symbol
        matching_dirs = []
        for agent_type in ['ppo', 'a2c']:
            pattern = f"{agent_type}_{symbol}_*"
            matching_dirs.extend(models_dir.glob(pattern))

        if not matching_dirs:
            return None

        # Sort by modification time (most recent first)
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Check for best_model.zip or final_model.zip
        latest_dir = matching_dirs[0]
        model_path = latest_dir / "best_model.zip"
        if not model_path.exists():
            model_path = latest_dir / "final_model.zip"

        if not model_path.exists():
            return None

        # Extract agent type and mode from directory name
        dir_name = latest_dir.name
        agent_type = 'ppo' if 'ppo' in dir_name.lower() else 'a2c'

        # Check if using improved actions by looking at config
        use_improved_actions = False
        config_path = latest_dir / "training_config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                use_improved_actions = config.get('use_improved_actions', False)

        return {
            'path': model_path,
            'agent_type': agent_type,
            'directory': latest_dir,
            'use_improved_actions': use_improved_actions
        }

    def _run_backtest(self, event):
        """Run backtest comparison."""
        self.results_panel.clear()
        self.results_panel.append(pn.indicators.LoadingSpinner(value=True, size=50))
        self.model_status_pane.object = ""

        symbol = self.symbol_input.value.strip().upper()
        if not symbol:
            pn.state.notifications.error("Please enter a stock symbol", duration=3000)
            return

        pn.state.notifications.info(f"Running backtest on {symbol}...", duration=3000)

        def backtest_thread():
            try:
                from src.rl import BacktestEngine, BacktestConfig
                from src.rl.training import EnhancedRLTrainer
                from src.rl.baselines import BuyHoldStrategy, MomentumStrategy

                # Setup dates (last 6 months)
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

                config = BacktestConfig(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )

                engine = BacktestEngine(config)
                engine.setup_environment()

                # Run strategies
                results = {}

                # Check if there's a trained RL model for this symbol
                model_info = self._find_latest_model(symbol)
                if model_info:
                    try:
                        # Load the trained agent
                        agent = EnhancedRLTrainer.load_agent(
                            model_path=model_info['path'],
                            agent_type=model_info['agent_type'],
                            env=engine.env
                        )

                        # Determine agent display name
                        agent_name = f"{model_info['agent_type'].upper()} Agent"

                        # Run backtest
                        results[agent_name] = engine.run_agent_backtest(agent, deterministic=True)

                        # Update model status
                        model_name = model_info['directory'].name
                        status_html = f"""<div style='padding: 10px; background: #D1FAE5;
                                          border-radius: 4px; font-size: 12px; color: #065F46;
                                          border: 1px solid #A7F3D0;'>
                            ✅ Using model: <strong>{agent_name}</strong><br>
                            Directory: {model_name}
                        </div>"""
                        pn.state.execute(lambda: setattr(self.model_status_pane, 'object', status_html))

                        pn.state.execute(lambda: pn.state.notifications.info(
                            f"Loaded {agent_name}",
                            duration=3000
                        ))
                    except Exception as e:
                        logger.error(f"Error loading RL agent: {e}", exc_info=True)
                        pn.state.execute(lambda: pn.state.notifications.warning(
                            f"Could not load RL agent: {str(e)}",
                            duration=4000
                        ))

                # Run baseline strategies
                buy_hold = BuyHoldStrategy()
                results['Buy & Hold'] = engine.run_strategy_backtest(buy_hold.get_action)

                momentum = MomentumStrategy()
                results['Momentum'] = engine.run_strategy_backtest(momentum.get_action)

                # Display results
                pn.state.execute(lambda: self._display_backtest_results(
                    results, symbol, model_info.get('use_improved_actions', False) if model_info else False
                ))
                pn.state.execute(lambda: pn.state.notifications.success("Backtest complete!", duration=3000))

            except Exception as e:
                logger.error(f"Backtest error: {e}", exc_info=True)
                pn.state.execute(lambda: self.results_panel.clear())
                pn.state.execute(lambda: pn.state.notifications.error(
                    f"Backtest failed: {str(e)}", duration=5000
                ))

        thread = threading.Thread(target=backtest_thread)
        thread.daemon = True
        thread.start()

    def _display_backtest_results(self, results: Dict, symbol: str, use_improved_actions: bool = False):
        """Display backtest results."""
        self.results_panel.clear()

        # Header
        header_html = f"""
        <div style='background: #F8F9FA;
                    border: 1px solid #DEE2E6;
                    border-left: 4px solid #0891B2;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h3 style='margin: 0; color: #212529;'>📊 Backtest Results: {symbol}</h3>
            <p style='margin: 5px 0 0 0; color: #495057; font-size: 13px;'>
                Last 6 months performance comparison
            </p>
        </div>
        """
        self.results_panel.append(pn.pane.HTML(header_html))

        # Action names based on action space
        if use_improved_actions:
            action_names = {
                0: 'HOLD', 1: 'BUY_S', 2: 'BUY_M', 3: 'BUY_L', 4: 'SELL_P', 5: 'SELL_A'
            }
            action_order = ['HOLD', 'BUY_S', 'BUY_M', 'BUY_L', 'SELL_P', 'SELL_A']
        else:
            action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY_S', 3: 'BUY_L'}
            action_order = ['SELL', 'HOLD', 'BUY_S', 'BUY_L']

        # Combined metrics and action distribution table
        combined_html = "<div style='overflow-x: auto;'><table style='width: 100%; border-collapse: collapse; font-size: 13px;'>"
        combined_html += "<thead><tr style='background: #f3f4f6;'>"
        combined_html += "<th style='padding: 10px; text-align: left; border: 1px solid #e5e7eb;'>Strategy</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Return</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Sharpe</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Max DD</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Win Rate</th>"

        # Add action columns
        for action_name in action_order:
            combined_html += f"<th style='padding: 10px; text-align: center; border: 1px solid #e5e7eb;'>{action_name}</th>"

        combined_html += "<th style='padding: 10px; text-align: center; border: 1px solid #e5e7eb;'>Executed</th>"
        combined_html += "</tr></thead><tbody>"

        for name, result in results.items():
            m = result.metrics
            return_color = '#10b981' if m.total_return_pct >= 0 else '#ef4444'

            # Count actions
            action_counts = {action_name: 0 for action_name in action_order}
            for action in result.actions:
                if isinstance(action, (np.ndarray, np.generic)):
                    action_scalar = int(action.item())
                else:
                    action_scalar = int(action)

                action_name = action_names.get(action_scalar, f'A{action_scalar}')
                if action_name in action_counts:
                    action_counts[action_name] += 1

            total_actions = sum(action_counts.values())
            total_trades = len(result.trades)

            combined_html += f"<tr style='border: 1px solid #e5e7eb;'>"
            combined_html += f"<td style='padding: 10px; font-weight: bold;'>{name}</td>"
            combined_html += f"<td style='padding: 10px; text-align: right; color: {return_color}; font-weight: bold;'>{m.total_return_pct:+.2f}%</td>"
            combined_html += f"<td style='padding: 10px; text-align: right;'>{m.sharpe_ratio:.2f}</td>"
            combined_html += f"<td style='padding: 10px; text-align: right; color: #ef4444;'>{abs(m.max_drawdown)*100:.2f}%</td>"
            combined_html += f"<td style='padding: 10px; text-align: right;'>{m.win_rate*100:.0f}%</td>"

            # Add action counts with colors
            for action_name in action_order:
                count = action_counts[action_name]
                pct = (count / total_actions * 100) if total_actions > 0 else 0

                # Color based on action type
                if 'SELL' in action_name:
                    color = '#ef4444'
                elif 'HOLD' in action_name:
                    color = '#6b7280'
                else:  # BUY variants
                    color = '#3b82f6'

                combined_html += f"<td style='padding: 10px; text-align: center;'>"
                combined_html += f"<span style='color: {color}; font-weight: bold;'>{count}</span> "
                combined_html += f"<span style='color: #9ca3af; font-size: 11px;'>({pct:.0f}%)</span>"
                combined_html += "</td>"

            combined_html += f"<td style='padding: 10px; text-align: center; font-weight: bold; color: #059669;'>{total_trades}</td>"
            combined_html += "</tr>"

        combined_html += "</tbody></table></div>"
        self.results_panel.append(pn.pane.HTML(combined_html))

        # Charts
        try:
            from src.rl import RLVisualizer

            # Strategy comparison
            fig = RLVisualizer.plot_strategy_comparison(results, title=f"Performance Comparison - {symbol}")
            self.results_panel.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))

            # Action comparison
            fig_actions = RLVisualizer.plot_action_comparison(results, title="Action Distribution Comparison")
            self.results_panel.append(pn.pane.Plotly(fig_actions, sizing_mode="stretch_width", height=400))

            # Metrics comparison
            fig2 = RLVisualizer.plot_metrics_comparison(results, title="Key Metrics")
            self.results_panel.append(pn.pane.Plotly(fig2, sizing_mode="stretch_width", height=300))

        except Exception as e:
            logger.error(f"Error creating charts: {e}", exc_info=True)

    def get_panel(self):
        """Get the main panel."""
        # Configuration section
        config_section = pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Symbol</div>"),
                    self.symbol_input,
                    width=150
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Algorithm</div>"),
                    self.agent_type,
                    width=180
                ),
                align='start',
                sizing_mode="stretch_width"
            ),
            pn.pane.HTML("<div style='font-size: 13px; font-weight: 600; color: #374151; margin: 10px 0 8px 0;'>Training Options</div>"),
            pn.pane.HTML("<div style='font-size: 11px; color: #6b7280; margin-bottom: 8px;'>✅ Action Masking and 6-Action Space are always enabled</div>"),
            pn.Row(
                pn.Column(self.use_enhanced_rewards, self.use_adaptive_sizing, width=250),
                pn.Column(self.use_curriculum, width=250),
            ),
            pn.Row(
                self.training_days,
                self.timesteps,
                sizing_mode="stretch_width"
            ),
            pn.pane.HTML("""
                <div style='font-size: 11px; color: #059669; background: #D1FAE5;
                            padding: 8px 12px; border-radius: 4px; margin: 8px 0;
                            border-left: 3px solid #059669;'>
                    <strong>💡 Proven Formula:</strong> 300k steps with 3 years of data consistently beats Buy & Hold.
                    Reduce to 100k steps for quick testing.
                </div>
            """),
            pn.Row(
                self.ent_coef,
                sizing_mode="stretch_width"
            ),
            pn.Row(
                self.train_button,
                self.backtest_button,
                sizing_mode="stretch_width"
            ),
            self.progress_bar,
            self.model_status_pane,
            styles=dict(background='#F8F9FA', border_radius='8px', padding='15px'),
            margin=(0, 0, 15, 0)
        )

        return pn.Column(
            config_section,
            self.results_panel,
            pn.pane.HTML(HTMLComponents.disclaimer()),
            sizing_mode="stretch_width"
        )


def create_rl_training_panel():
    """Create and return the RL training panel."""
    panel = RLTrainingPanel()
    return panel.get_panel()
