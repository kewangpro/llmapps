"""
Models Page - LSTM and RL Model Registry
"""

import panel as pn
import param
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from src.ui.design_system import Colors, HTMLComponents, TableStyles

logger = logging.getLogger(__name__)


class ModelsPage(param.Parameterized):
    """Model registry and management"""

    def __init__(self, **params):
        super().__init__(**params)
        self._create_ui()

    def _create_ui(self):
        """Create models UI"""
        self.models_panel = pn.Column(sizing_mode="stretch_width")
        self.refresh_button = pn.widgets.Button(
            name="🔄 Refresh",
            button_type="primary",
            width=120
        )
        self.refresh_button.on_click(self._load_models)

        # Auto-load on init
        self._load_models()

    def _load_models(self, event=None):
        """Load model registry"""
        self.models_panel.clear()

        # LSTM Models Section
        self.models_panel.append(pn.pane.HTML(
            HTMLComponents.section_header("🧠 LSTM Models", "Trained prediction models")
        ))

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
            self.models_panel.append(pn.pane.HTML(table_html))
        else:
            self.models_panel.append(pn.pane.HTML(f"""
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

        # RL Models Section
        self.models_panel.append(pn.pane.HTML(
            HTMLComponents.section_header("🤖 RL Trading Agents", "Reinforcement learning models")
        ))

        rl_models = self._get_rl_models()

        if rl_models:
            headers = ["Agent Name", "Symbol", "Algorithm", "Trained", "Performance", "Actions"]
            rows = []

            for model in rl_models:
                perf_value = model.get('return', 'N/A')
                # Check if return is a number and set color accordingly
                if isinstance(perf_value, (int, float)):
                    perf_color = Colors.SUCCESS_GREEN if perf_value >= 0 else Colors.DANGER_RED
                    perf_display = f'{perf_value:+.2f}%'
                else:
                    perf_color = Colors.TEXT_SECONDARY
                    perf_display = '<span style="font-size: 0.7rem;">Run backtest →</span>'

                rows.append([
                    model['name'],
                    model['symbol'],
                    model['algorithm'],
                    model['trained'],
                    f'<span style="color: {perf_color};">{perf_display}</span>',
                    f'<button style="background: {Colors.ACCENT_PURPLE}; color: white; border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 0.75rem;">Load</button>'
                ])

            table_html = TableStyles.generate_table(headers, rows)
            self.models_panel.append(pn.pane.HTML(table_html))
        else:
            self.models_panel.append(pn.pane.HTML(f"""
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
                    'final_loss': final_loss,
                    'val_loss': val_loss,
                    'size': f'{len(keras_files)} models'
                }
                models.append(model_info)

        except Exception as e:
            logger.error(f"Error loading LSTM models: {e}")

        return models

    def _get_rl_models(self) -> List[Dict]:
        """Get list of RL models"""
        models = []
        models_dir = Path("data/models/rl")

        if not models_dir.exists():
            return models

        try:
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    model_file = model_dir / "final_model.zip"
                    if model_file.exists():
                        # Extract info from directory name (e.g., ppo_AAPL_20250102_123456)
                        parts = model_dir.name.split('_')
                        algorithm = parts[0].upper() if parts else "Unknown"
                        symbol = parts[1].upper() if len(parts) > 1 else "Unknown"

                        model_info = {
                            'name': model_dir.name,
                            'symbol': symbol,
                            'algorithm': algorithm,
                            'trained': datetime.fromtimestamp(model_dir.stat().st_mtime).strftime('%Y-%m-%d'),
                            'return': 'N/A'  # Would need backtest results
                        }
                        models.append(model_info)

        except Exception as e:
            logger.error(f"Error loading RL models: {e}")

        return models

    def get_view(self):
        """Get the models view"""
        return pn.Column(
            HTMLComponents.page_header(
                "Models",
                "LSTM and RL model registry"
            ),
            pn.Row(
                self.refresh_button,
                sizing_mode="stretch_width",
                margin=(0, 0, 15, 0)
            ),
            self.models_panel,
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )
