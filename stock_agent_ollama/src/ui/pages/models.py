"""
Models Page - LSTM and RL Model Registry
"""

import panel as pn
import param
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from collections import Counter

from src.ui.design_system import Colors, HTMLComponents, TableStyles

logger = logging.getLogger(__name__)


class ModelsPage(param.Parameterized):
    """Model registry and management"""

    # Reactive parameter to track selected models
    selected_models = param.List(default=[])

    def __init__(self, **params):
        super().__init__(**params)
        self._create_ui()

    def _create_ui(self):
        """Create models UI"""
        # Create separate panels for each tab
        self.lstm_panel = pn.Column(sizing_mode="stretch_width")
        self.rl_panel = pn.Column(sizing_mode="stretch_width")

        # Backtest button (initially hidden) - same style as Training page
        self.backtest_button = pn.widgets.Button(
            name="📊 Run Backtest",
            button_type="primary",
            width=None, # Changed from 150 to None for dynamic sizing
            height=40,
            visible=False
        )
        self.backtest_button.on_click(self._run_batch_backtest)

        # Backtest results panel
        self.backtest_results_panel = pn.Column(sizing_mode="stretch_width")

        # Store checkboxes and model info for RL models
        self.rl_checkboxes = {}
        self.rl_model_data = []

        # Create dynamic header that changes with tab selection
        self.header_pane = pn.Column(sizing_mode="stretch_width")
        self._update_header(0)  # Set initial header for LSTM tab

        # Create tabs
        self.tabs = pn.Tabs(
            ("🧠 LSTM Models", self._create_lstm_tab()),
            ("🤖 RL Agents", self._create_rl_tab()),
            dynamic=True,
            sizing_mode="stretch_width"
        )

        # Watch for tab changes to update header
        self.tabs.param.watch(self._on_tab_change, 'active')

        # Auto-load on init
        self._load_models()

    def _create_lstm_tab(self) -> pn.Column:
        """Create LSTM models tab"""
        return pn.Column(
            self.lstm_panel,
            sizing_mode="stretch_width"
        )

    def _create_rl_tab(self) -> pn.Column:
        """Create RL agents tab"""
        return pn.Column(
            self.rl_panel,
            self.backtest_results_panel,
            sizing_mode="stretch_width"
        )

    def _update_header(self, tab_index):
        """Update header based on active tab"""
        if tab_index == 0:
            self.header_pane.clear()
            self.header_pane.append(pn.pane.HTML(HTMLComponents.page_header(
                "LSTM Models",
                "Trained prediction models"
            )))
        else:
            # For RL tab, show header without button
            self.header_pane.clear()
            self.header_pane.append(pn.pane.HTML(HTMLComponents.page_header(
                "RL Trading Agents",
                "Reinforcement learning models"
            )))

    def _on_tab_change(self, event):
        """Handle tab change event"""
        self._update_header(event.new)

    def _load_models(self, event=None):
        """Load model registry"""
        self._load_lstm_models()
        self._load_rl_models()

    def _load_lstm_models(self):
        """Load LSTM models into tab"""
        self.lstm_panel.clear()

        lstm_models = self._get_lstm_models()

        if lstm_models:
            headers = ["Model Name", "Symbol", "Trained", "Final Loss", "Val Loss", "Size", "Actions"]
            rows = []

            for model in lstm_models:
                rows.append([
                    model['name'],
                    model['symbol'],
                    model['trained'],
                    model['final_loss'],
                    model['val_loss'],
                    model['size'],
                    f'<button style="background: {Colors.ACCENT_PURPLE}; color: white; border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 0.75rem;">View</button>'
                ])

            table_html = TableStyles.generate_table(headers, rows)
            self.lstm_panel.append(pn.pane.HTML(table_html))
        else:
            self.lstm_panel.append(pn.pane.HTML(f"""
                <div style='background: {Colors.BG_SECONDARY};
                            border: 1px solid {Colors.BORDER_SUBTLE};
                            border-radius: 8px;
                            padding: 30px;
                            text-align: center;'>
                    <p style='color: {Colors.TEXT_SECONDARY}; margin: 0;'>
                        📊 No LSTM models trained yet. Train your first model from the Analysis page!
                    </p>
                </div>
            """))

    def _load_rl_models(self):
        """Load RL models into tab with checkboxes"""
        self.rl_panel.clear()
        self.rl_checkboxes = {}
        self.rl_model_data = self._get_rl_models()

        if self.rl_model_data:
            # Create table with checkboxes using Row layout like LSTM table
            # Header row
            header_html = f"""
            <style>
            .rl-table-header {{
                width: 100%;
                display: table;
                border-collapse: collapse;
                background: #F3F4F6;
                border: 1px solid {Colors.BORDER_SUBTLE};
                border-bottom: 2px solid {Colors.BORDER_SUBTLE};
                table-layout: fixed;
            }}
            .rl-table-header th {{
                text-align: left;
                padding: 12px 16px;
                font-weight: 600;
                color: {Colors.TEXT_PRIMARY};
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.05em;
            }}
            .rl-table-header th:first-child {{
                width: 40px;
                text-align: center;
            }}
            .rl-table-header th:nth-child(2) {{
                width: 30%;
            }}
            .rl-table-header th:nth-child(3) {{
                width: 8%;
            }}
            .rl-table-header th:nth-child(4) {{
                width: 15%;
            }}
            .rl-table-header th:nth-child(5) {{
                width: 12%;
            }}
            .rl-table-header th:nth-child(6) {{
                width: 10%;
            }}
            .rl-table-header th:nth-child(7) {{
                width: 12%;
            }}
            .rl-table-header th:nth-child(8) {{
                width: 10%;
            }}
            </style>
            <table class="rl-table-header">
                <tr>
                    <th></th>
                    <th>Agent Name</th>
                    <th>Symbol</th>
                    <th>Algorithm</th>
                    <th>Trained</th>
                    <th>Timesteps</th>
                    <th>Learn Rate</th>
                    <th>Size</th>
                </tr>
            </table>
            """

            self.rl_panel.append(pn.pane.HTML(header_html, sizing_mode="stretch_width", margin=(0,0,0,0)))

            # Create rows using Panel Row components
            for i, model in enumerate(self.rl_model_data):
                # Create checkbox
                checkbox = pn.widgets.Checkbox(
                    name="",
                    value=False,
                    width=30,
                    height=30,
                    margin=(10, 0, 10, 10),
                    align='center'
                )
                checkbox.param.watch(self._on_checkbox_change, 'value')
                self.rl_checkboxes[i] = checkbox

                # Create row content
                algo_badge = f'<span style="background: {Colors.ACCENT_PURPLE}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block;">{model["algorithm"].replace("_", " ")}</span>'

                row_html = f"""
                <div style="display: flex; align-items: center; width: 100%;">
                    <table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
                        <tr>
                            <td style="width: 30%; padding: 12px 8px; font-weight: 500; color: {Colors.TEXT_PRIMARY}; font-size: 13px; word-wrap: break-word;">{model['name']}</td>
                            <td style="width: 8%; padding: 12px 8px; color: {Colors.TEXT_PRIMARY}; font-size: 13px;">{model['symbol']}</td>
                            <td style="width: 15%; padding: 12px 8px; font-size: 13px;">{algo_badge}</td>
                            <td style="width: 12%; padding: 12px 8px; color: {Colors.TEXT_SECONDARY}; font-size: 13px;">{model['trained']}</td>
                            <td style="width: 10%; padding: 12px 8px; color: {Colors.TEXT_PRIMARY}; font-size: 13px;">{model['timesteps']}</td>
                            <td style="width: 12%; padding: 12px 8px; color: {Colors.TEXT_PRIMARY}; font-size: 13px;">{model['learning_rate']}</td>
                            <td style="width: 10%; padding: 12px 8px; color: {Colors.TEXT_SECONDARY}; font-size: 13px;">{model['size']}</td>
                        </tr>
                    </table>
                </div>
                """

                row = pn.Row(
                    checkbox,
                    pn.pane.HTML(row_html, sizing_mode="stretch_width", margin=(0,0,0,0)),
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                    styles={
                        'background': 'white',
                        'border-left': f'1px solid {Colors.BORDER_SUBTLE}',
                        'border-right': f'1px solid {Colors.BORDER_SUBTLE}',
                        'border-bottom': f'1px solid {Colors.BORDER_SUBTLE}',
                        'display': 'flex',
                        'align-items': 'center',
                    }
                )
                self.rl_panel.append(row)

            # Add backtest button at the bottom of the table
            self.rl_panel.append(pn.Row(
                self.backtest_button,
                sizing_mode="stretch_width",
                margin=(15, 0, 0, 0)
            ))

        else:
            self.rl_panel.append(pn.pane.HTML(f"""
                <div style='background: {Colors.BG_SECONDARY};
                            border: 1px solid {Colors.BORDER_SUBTLE};
                            border-radius: 8px;
                            padding: 30px;
                            text-align: center;'>
                    <p style='color: {Colors.TEXT_SECONDARY}; margin: 0;'>
                        🤖 No RL agents trained yet. Train your first agent from the Trading page!
                    </p>
                </div>
            """))

    def _get_lstm_models(self) -> List[Dict]:
        """Get list of LSTM models"""
        models = []
        models_dir = Path("data/models/lstm")

        if not models_dir.exists():
            return models

        try:
            # Look for metadata.json files directly in the lstm directory
            import json
            metadata_files = list(models_dir.glob("*_metadata.json"))

            for metadata_file in metadata_files:
                # Extract symbol from filename (e.g., AAPL_metadata.json -> AAPL)
                symbol = metadata_file.stem.replace('_metadata', '')

                # Count associated keras model files
                keras_files = list(models_dir.glob(f"{symbol}_model_*.keras"))

                # Try to load metadata for performance metrics
                final_loss = 'N/A'
                val_loss = 'N/A'
                trained_date = datetime.fromtimestamp(metadata_file.stat().st_mtime).strftime('%Y-%m-%d')

                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                        # Get training date if available
                        if 'training_date' in metadata:
                            trained_date = metadata['training_date'][:10]  # Get YYYY-MM-DD part

                        # Extract performance metrics from training histories
                        if 'training_histories' in metadata and metadata['training_histories']:
                            # Calculate average final loss and val_loss across all models
                            final_losses = []
                            val_losses = []

                            for history in metadata['training_histories']:
                                if 'loss' in history and history['loss']:
                                    final_losses.append(history['loss'][-1])
                                if 'val_loss' in history and history['val_loss']:
                                    val_losses.append(history['val_loss'][-1])

                            if final_losses:
                                avg_final_loss = sum(final_losses) / len(final_losses)
                                final_loss = f"{avg_final_loss:.4f}"

                            if val_losses:
                                avg_val_loss = sum(val_losses) / len(val_losses)
                                val_loss = f"{avg_val_loss:.4f}"

                except Exception as e:
                    logger.warning(f"Could not load metadata for {symbol}: {e}")

                model_info = {
                    'name': f"{symbol} LSTM Ensemble",
                    'symbol': symbol.upper(),
                    'trained': trained_date,
                    'trained_timestamp': metadata_file.stat().st_mtime,  # For sorting
                    'final_loss': final_loss,
                    'val_loss': val_loss,
                    'size': f'{len(keras_files)} models'
                }
                models.append(model_info)

        except Exception as e:
            logger.error(f"Error loading LSTM models: {e}")

        # Sort by trained time (newest first)
        models.sort(key=lambda x: x.get('trained_timestamp', 0), reverse=True)

        return models

    def _get_rl_models(self) -> List[Dict]:
        """Get list of RL models"""
        models = []
        models_dir = Path("data/models/rl")

        if not models_dir.exists():
            return models

        try:
            import json
            import os

            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    model_file = model_dir / "final_model.zip"
                    ensemble_config = model_dir / "ensemble" / "ensemble_config.json"

                    # Check for either regular model or ensemble model
                    if model_file.exists() or ensemble_config.exists():
                        # Extract info from directory name
                        # Format: algorithm_SYMBOL_YYYYMMDD_HHMMSS
                        # Examples: ppo_AAPL_20231114_120000, recurrent_ppo_TEAM_20251202_193237
                        parts = model_dir.name.split('_')

                        # For recurrent_ppo, the algorithm is the first two parts
                        if len(parts) >= 2 and parts[0] == 'recurrent' and parts[1] == 'ppo':
                            algorithm = 'RECURRENT_PPO'
                            symbol = parts[2].upper() if len(parts) > 2 else "Unknown"
                        else:
                            algorithm = parts[0].upper() if parts else "Unknown"
                            symbol = parts[1].upper() if len(parts) > 1 else "Unknown"

                        # Initialize metrics
                        timesteps = 'N/A'
                        learning_rate = 'N/A'
                        size = 'N/A'

                        # Try to load training config
                        training_config = model_dir / "training_config.json"
                        if training_config.exists():
                            try:
                                with open(training_config, 'r') as f:
                                    config_data = json.load(f)
                                    # Get total_timesteps
                                    if 'total_timesteps' in config_data:
                                        total_steps = config_data['total_timesteps']
                                        if total_steps >= 1_000_000:
                                            timesteps = f"{total_steps / 1_000_000:.1f}M"
                                        elif total_steps >= 1_000:
                                            timesteps = f"{total_steps / 1_000:.0f}K"
                                        else:
                                            timesteps = str(total_steps)
                                    # Get learning_rate
                                    if 'learning_rate' in config_data:
                                        lr = config_data['learning_rate']
                                        learning_rate = f"{lr:.4f}"
                            except Exception as e:
                                logger.debug(f"Could not load training config for {model_dir.name}: {e}")

                        # Try to load training log for timesteps if not found
                        training_log = model_dir / "training_log.json"
                        if training_log.exists() and timesteps == 'N/A':
                            try:
                                with open(training_log, 'r') as f:
                                    log_data = json.load(f)
                                    # Check for timesteps if not already found
                                    if 'timesteps' in log_data:
                                        total_steps = log_data['timesteps']
                                        if total_steps >= 1_000_000:
                                            timesteps = f"{total_steps / 1_000_000:.1f}M"
                                        elif total_steps >= 1_000:
                                            timesteps = f"{total_steps / 1_000:.0f}K"
                                        else:
                                            timesteps = str(total_steps)
                            except Exception as e:
                                logger.debug(f"Could not load training log for {model_dir.name}: {e}")

                        # Calculate model size
                        try:
                            if ensemble_config.exists():
                                # For ensemble, sum sizes of both models
                                ensemble_dir = model_dir / "ensemble"
                                ppo_model = ensemble_dir / "ppo_model.zip"
                                recurrent_model = ensemble_dir / "recurrent_ppo_model.zip"
                                total_size = 0
                                if ppo_model.exists():
                                    total_size += ppo_model.stat().st_size
                                if recurrent_model.exists():
                                    total_size += recurrent_model.stat().st_size
                                size = f"{total_size / (1024 * 1024):.1f} MB"
                            elif model_file.exists():
                                file_size = model_file.stat().st_size
                                size = f"{file_size / (1024 * 1024):.1f} MB"
                        except Exception as e:
                            logger.debug(f"Could not calculate size for {model_dir.name}: {e}")

                        model_info = {
                            'name': model_dir.name,
                            'directory': model_dir,
                            'symbol': symbol,
                            'algorithm': algorithm,
                            'trained': datetime.fromtimestamp(model_dir.stat().st_mtime).strftime('%Y-%m-%d'),
                            'trained_timestamp': model_dir.stat().st_mtime,  # For sorting
                            'timesteps': timesteps,
                            'learning_rate': learning_rate,
                            'size': size,
                            'return': 'N/A'  # Would need backtest results
                        }
                        models.append(model_info)

        except Exception as e:
            logger.error(f"Error loading RL models: {e}")

        # Sort by trained time (newest first)
        models.sort(key=lambda x: x.get('trained_timestamp', 0), reverse=True)

        return models

    def _on_checkbox_change(self, event):
        """Handle checkbox state change"""
        # Count selected models
        selected_count = sum(1 for cb in self.rl_checkboxes.values() if cb.value)

        # Show/hide backtest button based on selection
        self.backtest_button.visible = selected_count > 0

        # Update button text
        if selected_count > 0:
            self.backtest_button.name = f"Run Backtest ({selected_count} model{'s' if selected_count > 1 else ''} selected)"
        else:
            self.backtest_button.name = "Run Backtest"

    def _run_batch_backtest(self, event):
        """Run backtest for all selected models (supports multiple symbols)"""
        # Get selected models
        selected_indices = [i for i, cb in self.rl_checkboxes.items() if cb.value]

        if not selected_indices:
            pn.state.notifications.warning("No models selected", duration=3000)
            return

        selected_models = [self.rl_model_data[i] for i in selected_indices]

        # Group by symbol
        symbols_dict = {}
        for model in selected_models:
            symbol = model['symbol']
            if symbol not in symbols_dict:
                symbols_dict[symbol] = []
            symbols_dict[symbol].append(model)

        # Clear results and show loading
        self.backtest_results_panel.clear()
        self.backtest_results_panel.append(pn.indicators.LoadingSpinner(value=True, size=50))

        # Info about selected models
        total_models = len(selected_models)
        total_symbols = len(symbols_dict)
        pn.state.notifications.info(
            f"Running backtest on {total_models} model(s) across {total_symbols} symbol(s)...",
            duration=3000
        )

        def backtest_thread():
            try:
                from src.rl import BacktestEngine, BacktestConfig
                from src.rl.training import EnhancedRLTrainer
                from src.rl.baselines import BuyHoldStrategy, MomentumStrategy
                from src.rl.model_utils import load_env_config_from_model

                # Setup dates (last 6 months + lookback buffer)
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=280)).strftime("%Y-%m-%d")

                # Store results per symbol
                all_results = {}

                # Process each symbol separately
                for symbol, models_for_symbol in symbols_dict.items():
                    logger.info(f"Backtesting {len(models_for_symbol)} models for {symbol}")
                    results = {}

                    # Run backtest for each model in this symbol
                    for model in models_for_symbol:
                        model_dir = model['directory']
                        # Convert algorithm name to internal format
                        # RECURRENT_PPO -> recurrent_ppo, PPO -> ppo, ENSEMBLE -> ensemble
                        agent_type = model['algorithm'].lower()

                        # Find model path
                        model_file = model_dir / "final_model.zip"
                        ensemble_config = model_dir / "ensemble" / "ensemble_config.json"

                        if ensemble_config.exists():
                            model_path = model_dir / "ensemble"
                            agent_type = 'ensemble'
                        elif model_file.exists():
                            model_path = model_file
                        else:
                            logger.warning(f"No model file found for {model['name']}")
                            continue

                        try:
                            # Load training config
                            training_config = load_env_config_from_model(model_path)

                            # Determine include_trend_indicators
                            include_trend = training_config.get('include_trend_indicators', False)
                            if agent_type in ['recurrent_ppo', 'ensemble']:
                                include_trend = True

                            # Create backtest config
                            agent_backtest_config = BacktestConfig(
                                symbol=symbol,
                                start_date=start_date,
                                end_date=end_date,
                                use_improved_actions=training_config.get('use_improved_actions', True),
                                include_trend_indicators=include_trend,
                                # Inherit improvements from training config (default to True if not specified)
                                use_risk_manager=training_config.get('use_risk_manager', True),
                                use_adaptive_sizing=training_config.get('use_adaptive_sizing', True),
                                use_regime_detector=training_config.get('use_regime_detector', True),
                                use_mtf_features=training_config.get('use_mtf_features', True),
                                use_kelly_sizing=training_config.get('use_kelly_sizing', True)
                            )

                            # Create engine
                            engine = BacktestEngine(agent_backtest_config)

                            # Load agent
                            agent = EnhancedRLTrainer.load_agent(
                                model_path=model_path,
                                agent_type=agent_type,
                                env=None
                            )

                            # Run backtest
                            # Use symbol prefix for multi-symbol support
                            # Include unique model identifier to avoid duplicate keys
                            model_name_without_symbol = model['name'].replace(f"_{symbol}_", "_", 1)
                            agent_name = f"[{symbol}] {model_name_without_symbol}"
                            result = engine.run_agent_backtest(agent, deterministic=True)
                            results[agent_name] = result

                            # Save result to disk for validation/persistence
                            try:
                                result.save_to_model_dir(model_dir, agent_type)
                                logger.info(f"Saved backtest results for {agent_name}")
                            except Exception as e:
                                logger.error(f"Failed to save results for {agent_name}: {e}")

                            logger.info(f"Successfully backtested {agent_name}")

                        except Exception as e:
                            logger.error(f"Error backtesting {model['name']}: {e}", exc_info=True)

                    # Store results for this symbol (no baselines for batch backtest)
                    all_results[symbol] = results

                # Display all results
                pn.state.execute(lambda: self._display_multi_symbol_backtest_results(all_results, list(symbols_dict.keys())))

                # Show notification separately
                pn.state.execute(lambda: pn.state.notifications.success(
                    "✅ Backtest Complete! Results are displayed below.",
                    duration=5000
                ))

            except Exception as e:
                logger.error(f"Batch backtest error: {e}", exc_info=True)
                pn.state.execute(lambda: self.backtest_results_panel.clear())
                pn.state.execute(lambda: pn.state.notifications.error(
                    f"Backtest failed: {str(e)}", duration=5000
                ))

        thread = threading.Thread(target=backtest_thread)
        thread.daemon = True
        thread.start()

    def _display_batch_backtest_results(self, results: Dict, symbol: str):
        """Display batch backtest results with charts"""
        self.backtest_results_panel.clear()

        # Header with scroll target ID
        header_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-left: 4px solid {Colors.ACCENT_PURPLE};
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0 15px 0;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h3 style='margin: 0; color: {Colors.TEXT_PRIMARY};'>📊 Backtest Results: {symbol}</h3>
            <p style='margin: 5px 0 0 0; color: {Colors.TEXT_SECONDARY}; font-size: 13px;'>
                Last 6 months performance comparison
            </p>
        </div>
        """
        self.backtest_results_panel.append(pn.pane.HTML(header_html))

        # Action names (assume improved actions)
        action_names = {
            0: 'HOLD', 1: 'BUY_S', 2: 'BUY_M', 3: 'BUY_L', 4: 'SELL_P', 5: 'SELL_A'
        }
        action_order = ['HOLD', 'BUY_S', 'BUY_M', 'BUY_L', 'SELL_P', 'SELL_A']

        # Combined metrics and action distribution table
        combined_html = f"<div style='overflow-x: auto; margin-bottom: 20px;'><table style='width: 100%; border-collapse: collapse; font-size: 13px; background: white; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px;'>"
        combined_html += "<thead><tr style='background: #F3F4F6;'>"
        combined_html += f"<th style='padding: 10px; text-align: left; border: 1px solid {Colors.BORDER_SUBTLE};'>Strategy</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Return</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Sharpe</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Max DD</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Win Rate</th>"

        # Add action columns
        for action_name in action_order:
            combined_html += f"<th style='padding: 10px; text-align: center; border: 1px solid {Colors.BORDER_SUBTLE};'>{action_name}</th>"

        combined_html += f"<th style='padding: 10px; text-align: center; border: 1px solid {Colors.BORDER_SUBTLE};'>Executed</th>"
        combined_html += "</tr></thead><tbody>"

        for name, result in results.items():
            m = result.metrics
            return_color = Colors.SUCCESS_GREEN if m.total_return_pct >= 0 else Colors.DANGER_RED

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

            combined_html += f"<tr style='border: 1px solid {Colors.BORDER_SUBTLE};'>"
            combined_html += f"<td style='padding: 10px; font-weight: bold;'>{name}</td>"
            combined_html += f"<td style='padding: 10px; text-align: right; color: {return_color}; font-weight: bold;'>{m.total_return_pct:+.2f}%</td>"
            combined_html += f"<td style='padding: 10px; text-align: right;'>{m.sharpe_ratio:.2f}</td>"
            combined_html += f"<td style='padding: 10px; text-align: right; color: {Colors.DANGER_RED};'>{abs(m.max_drawdown)*100:.2f}%</td>"
            combined_html += f"<td style='padding: 10px; text-align: right;'>{m.win_rate*100:.0f}%</td>"

            # Add action counts with colors
            for action_name in action_order:
                count = action_counts[action_name]
                pct = (count / total_actions * 100) if total_actions > 0 else 0

                # Color based on action type
                if 'SELL' in action_name:
                    color = Colors.DANGER_RED
                elif 'HOLD' in action_name:
                    color = Colors.TEXT_SECONDARY
                else:  # BUY variants
                    color = Colors.ACCENT_PURPLE

                combined_html += f"<td style='padding: 10px; text-align: center;'>"
                combined_html += f"<span style='color: {color}; font-weight: bold;'>{count}</span> "
                combined_html += f"<span style='color: {Colors.TEXT_MUTED}; font-size: 11px;'>({pct:.0f}%)</span>"
                combined_html += "</td>"

            combined_html += f"<td style='padding: 10px; text-align: center; font-weight: bold; color: {Colors.SUCCESS_GREEN};'>{m.total_executed}</td>"
            combined_html += "</tr>"

        combined_html += "</tbody></table></div>"
        self.backtest_results_panel.append(pn.pane.HTML(combined_html))

        # Charts
        try:
            from src.rl import RLVisualizer

            # Strategy comparison
            fig = RLVisualizer.plot_strategy_comparison(results, title=f"Performance Comparison - {symbol}")
            self.backtest_results_panel.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))

            # Action comparison
            fig_actions = RLVisualizer.plot_action_comparison(results, title="Action Distribution Comparison")
            self.backtest_results_panel.append(pn.pane.Plotly(fig_actions, sizing_mode="stretch_width", height=400))

            # Metrics comparison
            fig2 = RLVisualizer.plot_metrics_comparison(results, title="Key Metrics")
            self.backtest_results_panel.append(pn.pane.Plotly(fig2, sizing_mode="stretch_width", height=300))

        except Exception as e:
            logger.error(f"Error creating charts: {e}", exc_info=True)

    def _display_multi_symbol_backtest_results(self, all_results: Dict[str, Dict], symbols: List[str]):
        """Display combined backtest results for multiple symbols"""
        self.backtest_results_panel.clear()

        # Header with scroll target ID
        symbols_str = ", ".join(symbols)
        header_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-left: 4px solid {Colors.ACCENT_PURPLE};
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0 15px 0;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h3 style='margin: 0; color: {Colors.TEXT_PRIMARY};'>📊 Backtest Results: {symbols_str}</h3>
            <p style='margin: 5px 0 0 0; color: {Colors.TEXT_SECONDARY}; font-size: 13px;'>
                Last 6 months performance comparison across {len(symbols)} symbol(s)
            </p>
        </div>
        """
        self.backtest_results_panel.append(pn.pane.HTML(header_html))

        # Combine all results into a single dictionary
        combined_results = {}
        for symbol, results in all_results.items():
            combined_results.update(results)

        # Display unified results
        self._display_unified_results(combined_results)

    def _display_unified_results(self, results: Dict):
        """Display unified results table and charts for all models"""
        # Action names (assume improved actions)
        action_names = {
            0: 'HOLD', 1: 'BUY_S', 2: 'BUY_M', 3: 'BUY_L', 4: 'SELL_P', 5: 'SELL_A'
        }
        action_order = ['HOLD', 'BUY_S', 'BUY_M', 'BUY_L', 'SELL_P', 'SELL_A']

        # Combined metrics and action distribution table
        combined_html = f"<div style='overflow-x: auto; margin-bottom: 20px;'><table style='width: 100%; border-collapse: collapse; font-size: 13px; background: white; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px;'>"
        combined_html += "<thead><tr style='background: #F3F4F6;'>"
        combined_html += f"<th style='padding: 10px; text-align: left; border: 1px solid {Colors.BORDER_SUBTLE};'>Model</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Return</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Sharpe</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Max DD</th>"
        combined_html += f"<th style='padding: 10px; text-align: right; border: 1px solid {Colors.BORDER_SUBTLE};'>Win Rate</th>"

        # Add action columns
        for action_name in action_order:
            combined_html += f"<th style='padding: 10px; text-align: center; border: 1px solid {Colors.BORDER_SUBTLE};'>{action_name}</th>"

        combined_html += f"<th style='padding: 10px; text-align: center; border: 1px solid {Colors.BORDER_SUBTLE};'>Executed</th>"
        combined_html += "</tr></thead><tbody>"

        for name, result in results.items():
            m = result.metrics
            return_color = Colors.SUCCESS_GREEN if m.total_return_pct >= 0 else Colors.DANGER_RED

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

            combined_html += f"<tr style='border: 1px solid {Colors.BORDER_SUBTLE};'>"
            combined_html += f"<td style='padding: 10px; font-weight: bold;'>{name}</td>"
            combined_html += f"<td style='padding: 10px; text-align: right; color: {return_color}; font-weight: bold;'>{m.total_return_pct:+.2f}%</td>"
            combined_html += f"<td style='padding: 10px; text-align: right;'>{m.sharpe_ratio:.2f}</td>"
            combined_html += f"<td style='padding: 10px; text-align: right; color: {Colors.DANGER_RED};'>{abs(m.max_drawdown)*100:.2f}%</td>"
            combined_html += f"<td style='padding: 10px; text-align: right;'>{m.win_rate*100:.0f}%</td>"

            # Add action counts with colors
            for action_name in action_order:
                count = action_counts[action_name]
                pct = (count / total_actions * 100) if total_actions > 0 else 0

                # Color based on action type
                if 'SELL' in action_name:
                    color = Colors.DANGER_RED
                elif 'HOLD' in action_name:
                    color = Colors.TEXT_SECONDARY
                else:  # BUY variants
                    color = Colors.ACCENT_PURPLE

                combined_html += f"<td style='padding: 10px; text-align: center;'>"
                combined_html += f"<span style='color: {color}; font-weight: bold;'>{count}</span> "
                combined_html += f"<span style='color: {Colors.TEXT_MUTED}; font-size: 11px;'>({pct:.0f}%)</span>"
                combined_html += "</td>"

            combined_html += f"<td style='padding: 10px; text-align: center; font-weight: bold; color: {Colors.SUCCESS_GREEN};'>{m.total_executed}</td>"
            combined_html += "</tr>"

        combined_html += "</tbody></table></div>"
        self.backtest_results_panel.append(pn.pane.HTML(combined_html))

        # Unified charts for all models
        try:
            from src.rl import RLVisualizer

            # Strategy comparison
            fig = RLVisualizer.plot_strategy_comparison(results, title="Performance Comparison - All Models")
            self.backtest_results_panel.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=400))

            # Action comparison
            fig_actions = RLVisualizer.plot_action_comparison(results, title="Action Distribution - All Models")
            self.backtest_results_panel.append(pn.pane.Plotly(fig_actions, sizing_mode="stretch_width", height=450))

            # Metrics comparison
            fig2 = RLVisualizer.plot_metrics_comparison(results, title="Key Metrics - All Models")
            self.backtest_results_panel.append(pn.pane.Plotly(fig2, sizing_mode="stretch_width", height=350))

        except Exception as e:
            logger.error(f"Error creating charts: {e}", exc_info=True)

    def get_view(self):
        """Get the models view"""
        return pn.Column(
            self.header_pane,
            self.tabs,
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )
