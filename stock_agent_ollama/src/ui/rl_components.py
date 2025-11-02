"""
Compact Panel UI components for RL trading features.
"""

import panel as pn
import param
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import threading

logger = logging.getLogger(__name__)


class CompactRLPanel(param.Parameterized):
    """Compact integrated RL panel for training and backtesting."""

    # Common parameters
    symbol = param.Selector(default="AAPL", objects=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "ORCL"])

    def __init__(self, **params):
        super().__init__(**params)
        self.trainer = None
        self.is_training = False
        self._create_ui()

    def _create_ui(self):
        """Create compact UI components."""
        # Training configuration (compact)
        self.agent_type = pn.widgets.RadioButtonGroup(
            name='Agent',
            options=['PPO', 'A2C'],
            value='PPO',
            button_type='primary',
            button_style='outline'
        )

        # LSTM feature extractor option
        self.use_lstm = pn.widgets.Checkbox(
            name='Use LSTM Features',
            value=False
        )

        self.training_days = pn.widgets.IntSlider(
            name='Training Period (days)',
            start=30,
            end=730,
            value=365,
            step=30,
            width=250
        )

        self.timesteps = pn.widgets.IntSlider(
            name='Training Steps',
            start=10000,
            end=100000,
            value=50000,
            step=10000,
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

        # Results panel
        self.results_panel = pn.Column(sizing_mode="stretch_width")

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

        pn.state.notifications.info(f"Starting {self.agent_type.value} training on {self.symbol}...", duration=3000)

        def train_thread():
            try:
                from src.rl import RLTrainer, TrainingConfig

                # Setup dates
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=self.training_days.value)).strftime("%Y-%m-%d")

                # Create config
                config = TrainingConfig(
                    symbol=self.symbol,
                    start_date=start_date,
                    end_date=end_date,
                    agent_type=self.agent_type.value.lower(),
                    total_timesteps=self.timesteps.value,
                    use_lstm=self.use_lstm.value,
                    verbose=1
                )

                # Train
                self.trainer = RLTrainer(config, progress_callback=self._update_progress)
                results = self.trainer.train()

                # Display results
                pn.state.execute(lambda: self._display_training_results(results))
                pn.state.execute(lambda: pn.state.notifications.success("Training complete!", duration=5000))

            except Exception as e:
                logger.error(f"Training error: {e}")
                pn.state.execute(lambda: pn.state.notifications.error(f"Training failed: {str(e)}", duration=5000))
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

    def _display_training_results(self, results: Dict):
        """Display training results - compact version."""
        self.results_panel.clear()

        # Summary card
        agent_name = self.agent_type.value
        if self.use_lstm.value:
            agent_name += "-LSTM"

        summary_html = f"""
        <div style='background: #F8F9FA;
                    border: 1px solid #DEE2E6;
                    border-left: 4px solid #0F9D58;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h3 style='margin: 0; color: #212529;'>✅ Training Complete</h3>
            <p style='margin: 5px 0 0 0; font-size: 0.9em; color: #495057;'>
                Agent: {agent_name} |
                Stock: {self.symbol} |
                Episodes: {results.get('total_episodes', 0)} |
                Time: {results.get('training_time', 0):.1f}s |
                Model: {Path(results.get('final_model_path', '')).name}
            </p>
        </div>
        """
        self.results_panel.append(pn.pane.HTML(summary_html))

        # Training progress chart
        if 'training_stats' in results:
            try:
                from src.rl import RLVisualizer
                fig = RLVisualizer.plot_training_progress(
                    results['training_stats'],
                    title=f"{agent_name} Training Progress - {self.symbol}"
                )
                self.results_panel.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))
            except Exception as e:
                logger.error(f"Error plotting training progress: {e}", exc_info=True)
                self.results_panel.append(pn.pane.Alert(
                    f"**Chart Error:** Could not create training progress chart: {str(e)}",
                    alert_type="warning"
                ))

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

        # Check if final_model.zip exists in the most recent directory
        latest_dir = matching_dirs[0]
        model_path = latest_dir / "final_model.zip"

        if not model_path.exists():
            return None

        # Extract agent type from directory name
        dir_name = latest_dir.name
        agent_type = 'ppo' if dir_name.startswith('ppo_') else 'a2c'

        return {
            'path': model_path,
            'agent_type': agent_type,
            'directory': latest_dir
        }

    def _run_backtest(self, event):
        """Run backtest comparison."""
        self.results_panel.clear()
        self.results_panel.append(pn.indicators.LoadingSpinner(value=True, size=50))

        pn.state.notifications.info(f"Running backtest on {self.symbol}...", duration=3000)

        def backtest_thread():
            try:
                from src.rl import BacktestEngine, BacktestConfig, RLTrainer
                from src.rl.baselines import BuyHoldStrategy, MomentumStrategy

                # Setup dates (last 6 months)
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

                config = BacktestConfig(
                    symbol=self.symbol,
                    start_date=start_date,
                    end_date=end_date
                )

                engine = BacktestEngine(config)
                engine.setup_environment()

                # Run strategies
                results = {}

                # Check if there's a trained RL model for this symbol
                model_info = self._find_latest_model(self.symbol)
                if model_info:
                    try:
                        # Load the trained agent
                        agent = RLTrainer.load_agent(
                            model_path=model_info['path'],
                            agent_type=model_info['agent_type'],
                            env=engine.env
                        )

                        # Run backtest with the RL agent (deterministic mode)
                        agent_name = f"{model_info['agent_type'].upper()} Agent"
                        results[agent_name] = engine.run_agent_backtest(agent, deterministic=True)

                        pn.state.execute(lambda: pn.state.notifications.info(
                            f"Loaded trained {model_info['agent_type'].upper()} model",
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
                pn.state.execute(lambda: self._display_backtest_results(results))
                pn.state.execute(lambda: pn.state.notifications.success("Backtest complete!", duration=3000))

            except Exception as e:
                logger.error(f"Backtest error: {e}")
                pn.state.execute(lambda: self.results_panel.clear())
                pn.state.execute(lambda: pn.state.notifications.error(f"Backtest failed: {str(e)}", duration=5000))

        thread = threading.Thread(target=backtest_thread)
        thread.daemon = True
        thread.start()

    def _display_backtest_results(self, results: Dict):
        """Display backtest results - compact."""
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
            <h3 style='margin: 0; color: #212529;'>📊 Backtest Results: {self.symbol}</h3>
            <p style='margin: 5px 0 0 0; color: #495057; font-size: 13px;'>Last 6 months performance comparison</p>
        </div>
        """
        self.results_panel.append(pn.pane.HTML(header_html))

        # Combined Metrics and Action Distribution Table
        action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY_SMALL', 3: 'BUY_LARGE'}

        combined_html = "<div style='overflow-x: auto;'><table style='width: 100%; border-collapse: collapse; font-size: 13px;'>"
        combined_html += "<thead><tr style='background: #f3f4f6;'>"
        combined_html += "<th style='padding: 10px; text-align: left; border: 1px solid #e5e7eb;'>Strategy</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Return</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Sharpe</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Max DD</th>"
        combined_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Win Rate</th>"
        combined_html += "<th style='padding: 10px; text-align: center; border: 1px solid #e5e7eb;'>SELL</th>"
        combined_html += "<th style='padding: 10px; text-align: center; border: 1px solid #e5e7eb;'>HOLD</th>"
        combined_html += "<th style='padding: 10px; text-align: center; border: 1px solid #e5e7eb;'>BUY_S</th>"
        combined_html += "<th style='padding: 10px; text-align: center; border: 1px solid #e5e7eb;'>BUY_L</th>"
        combined_html += "<th style='padding: 10px; text-align: center; border: 1px solid #e5e7eb;'>Executed</th>"
        combined_html += "</tr></thead><tbody>"

        for name, result in results.items():
            m = result.metrics
            return_color = '#10b981' if m.total_return_pct >= 0 else '#ef4444'

            # Count actions
            action_counts = {action_name: 0 for action_name in action_names.values()}
            for action in result.actions:
                action_name = action_names.get(action, f'Action {action}')
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
            for action_name in ['SELL', 'HOLD', 'BUY_SMALL', 'BUY_LARGE']:
                count = action_counts[action_name]
                pct = (count / total_actions * 100) if total_actions > 0 else 0

                # Color based on action type
                if action_name == 'SELL':
                    color = '#ef4444'
                elif action_name == 'HOLD':
                    color = '#6b7280'
                elif action_name == 'BUY_SMALL':
                    color = '#60a5fa'
                else:  # BUY_LARGE
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

            # Strategy comparison chart
            try:
                fig = RLVisualizer.plot_strategy_comparison(results, title=f"Performance Comparison - {self.symbol}")
                self.results_panel.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))
            except Exception as e:
                logger.error(f"Error plotting strategy comparison: {e}", exc_info=True)
                self.results_panel.append(pn.pane.Alert(
                    f"**Chart Error:** Could not create strategy comparison chart: {str(e)}",
                    alert_type="warning"
                ))

            # Action comparison chart
            try:
                fig_actions = RLVisualizer.plot_action_comparison(results, title="Action Distribution Comparison")
                self.results_panel.append(pn.pane.Plotly(fig_actions, sizing_mode="stretch_width", height=400))
            except Exception as e:
                logger.error(f"Error plotting action comparison: {e}", exc_info=True)
                self.results_panel.append(pn.pane.Alert(
                    f"**Chart Error:** Could not create action comparison chart: {str(e)}",
                    alert_type="warning"
                ))

            # Metrics comparison chart
            try:
                fig2 = RLVisualizer.plot_metrics_comparison(results, title="Key Metrics")
                self.results_panel.append(pn.pane.Plotly(fig2, sizing_mode="stretch_width", height=300))
            except Exception as e:
                logger.error(f"Error plotting metrics comparison: {e}", exc_info=True)
                self.results_panel.append(pn.pane.Alert(
                    f"**Chart Error:** Could not create metrics chart: {str(e)}",
                    alert_type="warning"
                ))

        except Exception as e:
            logger.error(f"Error importing visualizer: {e}", exc_info=True)
            self.results_panel.append(pn.pane.Alert(
                f"**Visualization Error:** {str(e)}",
                alert_type="danger"
            ))

    def get_panel(self):
        """Get the main panel."""
        # Simplified configuration layout with stock selector
        config_section = pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Symbol</div>"),
                    pn.widgets.Select.from_param(self.param.symbol, name='', width=120, height=35),
                    width=150
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Algorithm</div>"),
                    self.agent_type,
                    width=200
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Features</div>"),
                    self.use_lstm,
                    width=180
                ),
                align='start',
                sizing_mode="stretch_width"
            ),
            pn.Row(
                self.training_days,
                self.timesteps,
                sizing_mode="stretch_width"
            ),
            pn.Row(
                self.train_button,
                self.backtest_button,
                sizing_mode="stretch_width"
            ),
            self.progress_bar,
            styles=dict(background='#F8F9FA', border_radius='8px', padding='15px'),
            margin=(0, 0, 15, 0)
        )

        disclaimer_html = """
        <div style='background: #F8F9FA;
                    border-top: 1px solid #DEE2E6;
                    padding: 12px 20px;
                    text-align: center;
                    font-size: 11px;
                    color: #6C757D;
                    margin-top: 20px;'>
            ⚠️ <strong>Educational Disclaimer:</strong> For educational purposes only. Not financial advice. RL training typically takes 5-10 minutes for 50k steps. Past performance does not guarantee future results. Always consult qualified financial professionals before making investment decisions.
        </div>
        """

        return pn.Column(
            config_section,
            self.results_panel,
            pn.layout.VSpacer(),
            pn.pane.HTML(disclaimer_html),
            sizing_mode="stretch_both"
        )


def create_compact_rl_panel():
    """Create and return the compact RL panel."""
    panel = CompactRLPanel()
    return panel.get_panel()
