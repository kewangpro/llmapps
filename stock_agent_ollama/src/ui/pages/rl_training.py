"""
RL Training UI - Enhanced training with comprehensive improvements.

This panel provides:
- Enhanced RL training with action masking, curriculum learning, etc.
- Backtesting with support for improved action spaces

ARCHITECTURE NOTE - Single Source of Truth:
- All default configuration values are defined in dataclasses:
  * EnhancedTrainingConfig (src/rl/training.py)
  * EnhancedRewardConfig (src/rl/improvements.py)
- The UI reads these defaults and uses them for widget initialization
- DO NOT hardcode configuration values in the UI
- To change defaults, modify the dataclass definitions, not the UI
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
from src.tools.portfolio_manager import portfolio_manager

logger = logging.getLogger(__name__)


class RLTrainingPanel(param.Parameterized):
    """RL training panel with enhanced training system."""

    def __init__(self, **params):
        super().__init__(**params)
        self.trainer = None
        self.is_training = False

        # Load default configurations (single source of truth)
        from src.rl.training import EnhancedTrainingConfig
        from src.rl.improvements import EnhancedRewardConfig

        self._default_training_config = EnhancedTrainingConfig.__dataclass_fields__
        self._default_reward_config = EnhancedRewardConfig.__dataclass_fields__

        self._create_ui()

    def _create_ui(self):
        """Create UI components."""
        # Load watchlist symbols dynamically
        watchlist_symbols = portfolio_manager.load_portfolio("default")

        # Check if watchlist is empty
        if not watchlist_symbols:
            logger.warning("Watchlist is empty")
            watchlist_symbols = []

        # Symbol input with autocomplete from watchlist
        self.symbol_input = pn.widgets.AutocompleteInput(
            name='',
            value=watchlist_symbols[0] if watchlist_symbols else '',
            options=watchlist_symbols,
            placeholder='Enter symbol...',
            case_sensitive=False,
            restrict=False,  # Allow any symbol, not just from watchlist
            width=120,
            height=35,
            min_characters=1
        )

        # Agent type - 3 algorithms
        self.agent_type = pn.widgets.RadioButtonGroup(
            name='Algorithm',
            options=['PPO', 'RecurrentPPO', 'Ensemble'],
            value='PPO',
            button_type='primary',
            button_style='outline',
            width=450,
            height=40
        )

        # === TRAINING PARAMETERS ===
        # Note: All architectural improvements are always enabled:
        # - Action Masking (safety)
        # - 6-Action Space (improved actions)
        # - Enhanced Rewards (better learning signals)
        # - Adaptive Sizing (intelligent position sizing)
        # - Curriculum Learning is DISABLED (causes excessive invalid actions)

        # Training parameters
        self.training_days = pn.widgets.IntSlider(
            name='Training Period (days)',
            start=180,
            end=1095,
            value=1095,
            step=30,
            width=250
        )

        # Use dataclass defaults for widget values (single source of truth)
        self.timesteps = pn.widgets.IntSlider(
            name='Training Steps',
            start=50000,
            end=500000,
            value=self._default_training_config['total_timesteps'].default,
            step=10000,
            width=250
        )

        # Exploration bonus (entropy coefficient)
        self.ent_coef = pn.widgets.FloatSlider(
            name='Exploration Bonus (ent_coef)',
            start=0.0,
            end=0.1,
            value=self._default_training_config['ent_coef'].default,
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

    def refresh_symbol_dropdown(self):
        """Refresh symbol dropdown when watchlist changes"""
        try:
            watchlist_symbols = portfolio_manager.load_portfolio("default")
            if not watchlist_symbols:
                logger.warning("Watchlist is empty")
                watchlist_symbols = []
        except Exception as e:
            logger.error(f"Failed to load watchlist for symbol dropdown: {e}")
            watchlist_symbols = []

        # Update the options
        self.symbol_input.options = watchlist_symbols
        logger.info(f"Training page symbol dropdown refreshed with {len(watchlist_symbols)} stocks")

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

                # Use default reward config
                # EnhancedRLTrainer will automatically select:
                # - PPORewardConfig for PPO (stronger penalties to fight action collapse)
                # - RecurrentPPORewardConfig for RecurrentPPO (trend-following)
                # - Ensemble uses both PPO and RecurrentPPO configs
                reward_config = None  # Let trainer auto-select based on agent_type

                # Convert UI agent type to training format
                # UI: 'PPO', 'RecurrentPPO', 'Ensemble'
                # Training: 'ppo', 'recurrent_ppo', 'ensemble'
                agent_type_map = {
                    'PPO': 'ppo',
                    'RecurrentPPO': 'recurrent_ppo',
                    'Ensemble': 'ensemble'
                }
                agent_type = agent_type_map.get(self.agent_type.value, self.agent_type.value.lower())

                # Create training config (uses dataclass defaults for unspecified values)
                config = EnhancedTrainingConfig(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    agent_type=agent_type,
                    total_timesteps=self.timesteps.value,
                    ent_coef=self.ent_coef.value,
                    # transaction_cost_rate uses dataclass default (0.0005)

                    # All architectural improvements always enabled (not user-configurable)
                    use_action_masking=True,           # Safety - prevents invalid trades
                    use_enhanced_rewards=True,         # Better learning signals
                    use_adaptive_sizing=True,          # Intelligent position sizing
                    use_improved_actions=True,         # 6-action space
                    use_curriculum_learning=False,     # Disabled - causes excessive invalid actions
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

    def _find_latest_model(self, symbol: str, agent_type: str) -> Optional[Dict[str, Any]]:
        """Find the most recently trained model for a symbol and agent type."""
        models_dir = Path("data/models/rl")
        if not models_dir.exists():
            return None

        # Find all model directories for this symbol and agent type
        matching_dirs = []
        pattern = f"{agent_type.lower()}_{symbol}_*"
        matching_dirs.extend(models_dir.glob(pattern))

        if not matching_dirs:
            return None

        # Sort by modification time (most recent first)
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Only use completed models (those with final_model.zip or ensemble directory)
        # This prevents loading models that are still training
        for model_dir in matching_dirs:
            # Check for ensemble structure
            if agent_type.lower() == 'ensemble':
                ensemble_subdir = model_dir / "ensemble"
                if ensemble_subdir.exists() and (ensemble_subdir / "ppo_best_model.zip").exists():
                    latest_dir = model_dir
                    model_path = ensemble_subdir  # Point to ensemble subdirectory
                    break
            else:
                # Standard model structure
                final_model_path = model_dir / "final_model.zip"
                if final_model_path.exists():
                    latest_dir = model_dir
                    model_path = final_model_path
                    break
        else:
            # No completed models found
            return None

        # Check config for display info
        use_improved_actions = False
        config_path = latest_dir / "training_config.json"
        actual_agent_type = agent_type.lower()

        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                use_improved_actions = config.get('use_improved_actions', False)
                # Get actual agent type from config (handles legacy models)
                actual_agent_type = config.get('agent_type', agent_type).lower()

        # Determine display agent type
        display_agent_type = actual_agent_type.replace('_', ' ').title()

        return {
            'path': model_path,
            'agent_type': actual_agent_type,  # Actual type from config for loading
            'display_agent_type': display_agent_type,  # For display purposes
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
                from src.rl.model_utils import load_env_config_from_model

                # Setup dates (last 6 months + lookback buffer)
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=280)).strftime("%Y-%m-%d")

                results = {}
                loaded_models = []

                # Find and load ALL available trained RL models for this symbol
                for agent_type in ['ppo', 'recurrent_ppo', 'ensemble']:
                    logger.info(f"Backtest: Searching for {agent_type.upper()} model for {symbol}")
                    model_info = self._find_latest_model(symbol, agent_type)

                    if model_info:
                        try:
                            # Load training config to create the correct environment
                            training_config = load_env_config_from_model(model_info['path'])

                            # Determine include_trend_indicators
                            # RecurrentPPO ALWAYS uses trend indicators (13 features)
                            # Ensemble ALSO needs trend indicators (for its RecurrentPPO component)
                            include_trend = training_config.get('include_trend_indicators', False)
                            if agent_type in ['recurrent_ppo', 'ensemble']:
                                include_trend = True
                                logger.info(f"{agent_type.upper()} detected - forcing include_trend_indicators=True")

                            # Create a specific backtest config for this agent
                            agent_backtest_config = BacktestConfig(
                                symbol=symbol,
                                start_date=start_date,
                                end_date=end_date,
                                use_improved_actions=training_config.get('use_improved_actions', True),
                                include_trend_indicators=include_trend
                            )

                            # Create a new engine for this agent
                            engine = BacktestEngine(agent_backtest_config)

                            # Load the trained agent
                            agent = EnhancedRLTrainer.load_agent(
                                model_path=model_info['path'],
                                agent_type=model_info['agent_type'],
                                env=None  # Let the backtest engine create the env
                            )

                            # Determine agent display name
                            display_type = model_info.get('display_agent_type', model_info['agent_type'])
                            agent_name = f"{display_type.upper().replace('_', ' ')} Agent"

                            # Run backtest
                            results[agent_name] = engine.run_agent_backtest(agent, deterministic=True)

                            # Store agent name and model directory for status display
                            model_dir = model_info['directory'].name
                            loaded_models.append((agent_name, model_dir))

                            logger.info(f"Successfully loaded and backtested {agent_name}")

                        except Exception as e:
                            logger.error(f"Error loading {agent_type.upper()} agent: {e}", exc_info=True)

                # Run baseline strategies using a default engine
                default_config = BacktestConfig(symbol=symbol, start_date=start_date, end_date=end_date)
                baseline_engine = BacktestEngine(default_config)

                # Update model status with all loaded models
                if loaded_models:
                    models_list = "<br>".join([f"✅ {name} <span style='color: #059669; font-size: 11px;'>({dir_name})</span>"
                                               for name, dir_name in loaded_models])
                    status_html = f"""<div style='padding: 10px; background: #D1FAE5;
                                      border-radius: 4px; font-size: 12px; color: #065F46;
                                      border: 1px solid #A7F3D0;'>
                        <strong>Loaded Models:</strong><br>
                        {models_list}
                    </div>"""
                    pn.state.execute(lambda: setattr(self.model_status_pane, 'object', status_html))
                    pn.state.execute(lambda: pn.state.notifications.info(
                        f"Loaded {len(loaded_models)} model(s)", duration=3000
                    ))
                else:
                    pn.state.execute(lambda: pn.state.notifications.warning(
                        f"No trained models found for {symbol}", duration=4000
                    ))

                # Run baseline strategies
                buy_hold = BuyHoldStrategy()
                results['Buy & Hold'] = baseline_engine.run_strategy_backtest(buy_hold.get_action)

                momentum = MomentumStrategy()
                results['Momentum'] = baseline_engine.run_strategy_backtest(momentum.get_action)

                # Display results and show notification
                def show_results():
                    # All modern models use improved actions (6-action space)
                    # Default to True since line 217 enforces this for all training
                    self._display_backtest_results(results, symbol, use_improved_actions=True)
                    pn.state.notifications.success("✅ Backtest Complete! Results are displayed below.", duration=5000)
                
                pn.state.execute(show_results)

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

            # Price chart with buy/sell markers (RIGHT AFTER performance comparison)
            fig_price = RLVisualizer.plot_price_with_trades(results, symbol, title=f"{symbol} Price with Trade Signals")
            self.results_panel.append(pn.pane.Plotly(fig_price, sizing_mode="stretch_width", height=450))

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
                    width=450  # Wider to accommodate all 3 algorithm buttons
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
