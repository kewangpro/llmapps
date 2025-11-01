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
        summary_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 20px; border-radius: 8px; margin-bottom: 15px;'>
            <h3 style='margin: 0 0 15px 0;'>✅ Training Complete</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;'>
                <div>
                    <div style='opacity: 0.8; font-size: 12px;'>Agent</div>
                    <div style='font-size: 20px; font-weight: bold;'>{self.agent_type.value}</div>
                </div>
                <div>
                    <div style='opacity: 0.8; font-size: 12px;'>Stock</div>
                    <div style='font-size: 20px; font-weight: bold;'>{self.symbol}</div>
                </div>
                <div>
                    <div style='opacity: 0.8; font-size: 12px;'>Episodes</div>
                    <div style='font-size: 20px; font-weight: bold;'>{results.get('total_episodes', 0)}</div>
                </div>
                <div>
                    <div style='opacity: 0.8; font-size: 12px;'>Time</div>
                    <div style='font-size: 20px; font-weight: bold;'>{results.get('training_time', 0):.1f}s</div>
                </div>
            </div>
            <div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2);
                        font-size: 13px; opacity: 0.9;'>
                Model saved: {Path(results.get('final_model_path', '')).name}
            </div>
        </div>
        """
        self.results_panel.append(pn.pane.HTML(summary_html))

        # Training progress chart
        if 'training_stats' in results:
            try:
                from src.rl import RLVisualizer
                fig = RLVisualizer.plot_training_progress(
                    results['training_stats'],
                    title=f"{self.agent_type.value} Training Progress - {self.symbol}"
                )
                self.results_panel.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))
            except Exception as e:
                logger.error(f"Error plotting training progress: {e}", exc_info=True)
                self.results_panel.append(pn.pane.Alert(
                    f"**Chart Error:** Could not create training progress chart: {str(e)}",
                    alert_type="warning"
                ))

    def _run_backtest(self, event):
        """Run backtest comparison."""
        self.results_panel.clear()
        self.results_panel.append(pn.indicators.LoadingSpinner(value=True, size=50))

        pn.state.notifications.info(f"Running backtest on {self.symbol}...", duration=3000)

        def backtest_thread():
            try:
                from src.rl import BacktestEngine, BacktestConfig
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
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
            <h3 style='margin: 0;'>📊 Backtest Results: {self.symbol}</h3>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 13px;'>Last 6 months performance comparison</p>
        </div>
        """
        self.results_panel.append(pn.pane.HTML(header_html))

        # Metrics comparison table (compact)
        metrics_html = "<div style='overflow-x: auto;'><table style='width: 100%; border-collapse: collapse; font-size: 13px;'>"
        metrics_html += "<thead><tr style='background: #f3f4f6;'>"
        metrics_html += "<th style='padding: 10px; text-align: left; border: 1px solid #e5e7eb;'>Strategy</th>"
        metrics_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Return</th>"
        metrics_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Sharpe</th>"
        metrics_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Max DD</th>"
        metrics_html += "<th style='padding: 10px; text-align: right; border: 1px solid #e5e7eb;'>Win Rate</th>"
        metrics_html += "</tr></thead><tbody>"

        for name, result in results.items():
            m = result.metrics
            return_color = '#10b981' if m.total_return_pct >= 0 else '#ef4444'
            metrics_html += f"<tr style='border: 1px solid #e5e7eb;'>"
            metrics_html += f"<td style='padding: 10px; font-weight: bold;'>{name}</td>"
            metrics_html += f"<td style='padding: 10px; text-align: right; color: {return_color}; font-weight: bold;'>{m.total_return_pct:+.2f}%</td>"
            metrics_html += f"<td style='padding: 10px; text-align: right;'>{m.sharpe_ratio:.2f}</td>"
            metrics_html += f"<td style='padding: 10px; text-align: right; color: #ef4444;'>{abs(m.max_drawdown)*100:.2f}%</td>"
            metrics_html += f"<td style='padding: 10px; text-align: right;'>{m.win_rate*100:.0f}%</td>"
            metrics_html += "</tr>"

        metrics_html += "</tbody></table></div>"
        self.results_panel.append(pn.pane.HTML(metrics_html))

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
        # Configuration card (compact)
        config_card = pn.Card(
            pn.pane.HTML("<h4 style='margin: 0 0 10px 0;'>Configuration</h4>"),
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px;'>Stock Symbol</div>"),
                    pn.widgets.Select.from_param(self.param.symbol, width=120, height=35),
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px;'>Algorithm</div>"),
                    self.agent_type,
                ),
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
            title="⚙️ Settings",
            collapsed=False,
            collapsible=True,
            header_background='#f3f4f6',
            header_color='#374151',
            margin=(0, 0, 15, 0)
        )

        # Info card
        info_html = """
        <div style='background: #f0f9ff; border: 1px solid #bae6fd; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
            <h4 style='margin: 0 0 10px 0; color: #0369a1;'>ℹ️ About RL Trading</h4>
            <ul style='margin: 0; padding-left: 20px; font-size: 13px; color: #075985; line-height: 1.6;'>
                <li><strong>Training:</strong> Teach AI agents to learn profitable trading strategies</li>
                <li><strong>Backtesting:</strong> Compare strategies on historical data</li>
                <li><strong>PPO:</strong> Stable, sample-efficient (recommended)</li>
                <li><strong>A2C:</strong> Faster training, good for experimentation</li>
                <li><strong>Training time:</strong> 5-10 minutes for 50k steps</li>
            </ul>
            <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #bae6fd; font-size: 12px; color: #0c4a6e;'>
                ⚠️ <strong>Educational purpose only.</strong> Not financial advice. Past performance doesn't guarantee future results.
            </div>
        </div>
        """

        return pn.Column(
            pn.pane.HTML("<h3 style='margin: 0 0 15px 0; color: #374151;'>🤖 Reinforcement Learning Trading</h3>"),
            pn.pane.HTML(info_html),
            config_card,
            self.results_panel,
            sizing_mode="stretch_width"
        )


def create_compact_rl_panel():
    """Create and return the compact RL panel."""
    panel = CompactRLPanel()
    return panel.get_panel()
