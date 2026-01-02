import panel as pn
import param
import asyncio
import html
import json
from typing import Dict, Any, Optional, Tuple
import logging
import pandas as pd

from src.agents.query_processor import QueryProcessor
from src.agents.hybrid_query_processor import HybridQueryProcessor
from src.tools.visualizer import Visualizer
from src.tools.portfolio_manager import portfolio_manager # Import portfolio_manager

logger = logging.getLogger(__name__)

# Configure Panel
pn.extension('plotly', template='bootstrap', notifications=True)

class PandasEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle pandas Series and NaN values."""
    def default(self, obj):
        if isinstance(obj, pd.Series):
            return obj.where(pd.notna(obj), None).tolist()
        return super().default(obj)

def replace_nan_with_none(obj):
    """Recursively replace NaN values with None in nested data structures."""
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(i) for i in obj]
    elif isinstance(obj, float) and pd.isna(obj):
        return None
    return obj

class StockAnalysisApp(param.Parameterized):
    """Main Stock Analysis Panel Application with RL Integration"""

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize components - use enhanced processor with Ollama integration
        try:
            self.query_processor = HybridQueryProcessor()
            logger.debug("Using HybridQueryProcessor with Ollama integration")
        except Exception as e:
            logger.warning(f"Failed to initialize HybridQueryProcessor, falling back to legacy: {e}")
            self.query_processor = QueryProcessor()
        self.visualizer = Visualizer()

        # UI Components for Analysis Tab
        self.query_input = pn.widgets.TextInput(
            placeholder="Ask about stocks (e.g., 'Analyze AAPL' or 'Predict GOOGL')",
            sizing_mode="stretch_width",
            height=45,
        )

        self.submit_button = pn.widgets.Button(
            name="Analyze",
            button_type="primary",
            width=100,
            icon='search',
        )

        self.force_retrain_checkbox = pn.widgets.Checkbox(
            name="Force Retrain LSTM Model",
            value=False,
            margin=(5, 10),
        )

        self.loading_indicator = pn.indicators.LoadingSpinner(
            value=False,
            size=25,
            color='primary'
        )

        self.results_column = pn.Column(
            sizing_mode="stretch_width",
            min_height=300,
            max_height=700, # Constrain height to prevent excessive scrolling
            scroll=True, # Make the column itself scrollable
        )

        # LSTM Training progress tracking
        self.lstm_training_progress = pn.Column(visible=False)
        self.lstm_progress_bar = pn.indicators.Progress(
            name='LSTM Training Progress',
            value=0,
            max=100,
            width=300,
            bar_color='success'
        )
        self.lstm_status_text = pn.pane.HTML("")

        # LSTM Prediction progress tracking
        self.lstm_prediction_progress = pn.Column(visible=False)
        self.lstm_prediction_bar = pn.indicators.Progress(
            name='Generating Predictions',
            value=0,
            max=100,
            width=300,
            bar_color='info'
        )
        self.lstm_prediction_text = pn.pane.HTML("")

        # Quick action buttons - more compact with wrapping
        try:
            quick_stocks = portfolio_manager.load_portfolio("default")
            if not quick_stocks:
                logger.warning("Watchlist is empty")
                quick_stocks = []
        except Exception as e:
            logger.error(f"Failed to load watchlist for quick buttons: {e}")
            quick_stocks = []

        self.quick_buttons = pn.FlexBox(
            *[pn.widgets.Button(name=sym, button_type="light", width=70, height=30)
              for sym in quick_stocks],
            sizing_mode="stretch_width",
            flex_wrap="wrap"
        )

        # Bind events
        self.submit_button.on_click(self._handle_query)
        self.query_input.param.watch(self._on_enter_key, 'value')

        for button in self.quick_buttons:
            button.on_click(self._handle_quick_button)



    def refresh_quick_buttons(self):
        """Refresh quick action buttons when watchlist changes"""
        try:
            quick_stocks = portfolio_manager.load_portfolio("default")
            if not quick_stocks:
                logger.warning("Watchlist is empty")
                quick_stocks = []
        except Exception as e:
            logger.error(f"Failed to load watchlist for quick buttons: {e}")
            quick_stocks = []

        # Clear existing buttons
        self.quick_buttons.clear()

        # Create new buttons
        new_buttons = [pn.widgets.Button(name=sym, button_type="light", width=70, height=30)
                       for sym in quick_stocks]

        # Add new buttons to the container (will wrap as needed)
        self.quick_buttons.extend(new_buttons)

        # Rebind click events
        for button in self.quick_buttons:
            button.on_click(self._handle_quick_button)

        logger.info(f"Quick buttons refreshed with {len(quick_stocks)} stocks")

    def _on_enter_key(self, event):
        """Handle Enter key press"""
        pass  # Panel doesn't have native Enter support

    def _handle_quick_button(self, event):
        """Handle quick action button clicks"""
        symbol = event.obj.name
        # Use "Predict" if force retrain is checked, otherwise "Analyze"
        action = "Predict" if self.force_retrain_checkbox.value else "Analyze"
        self.query_input.value = f"{action} {symbol}"
        self._handle_query()

    def analyze_symbol(self, symbol: str, force_retrain: bool = False):
        """Programmatically analyze a symbol (for external triggers like watchlist clicks)"""
        self.force_retrain_checkbox.value = force_retrain
        action = "Predict" if force_retrain else "Analyze"
        self.query_input.value = f"{action} {symbol}"
        self._handle_query()

    def _update_prediction_progress(self, progress_data: Dict):
        """Update LSTM prediction progress."""
        progress_type = progress_data.get('type', '')

        if progress_type == 'prediction_progress':
            # Show progress panel if not visible
            if not self.lstm_prediction_progress.visible:
                pn.state.execute(lambda: setattr(self.lstm_prediction_progress, 'visible', True))

            symbol = progress_data.get('symbol', '')
            step = progress_data.get('step', '')
            progress = progress_data.get('progress', 0)

            # Update progress bar
            pn.state.execute(lambda: setattr(self.lstm_prediction_bar, 'value', progress))

            # Update status text
            from src.ui.design_system import Colors
            status_html = f"""
            <div style='background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                        color: white; padding: 12px; border-radius: 8px; margin: 10px 0;'>
                <h4 style='margin: 0 0 8px 0; font-size: 0.95rem;'>🔮 {step}</h4>
                <div style='font-size: 0.85rem; opacity: 0.9;'>{symbol} • {progress}% complete</div>
            </div>
            """
            pn.state.execute(lambda: setattr(self.lstm_prediction_text, 'object', status_html))

            # Hide immediately when complete
            if progress >= 100:
                pn.state.execute(lambda: setattr(self.lstm_prediction_progress, 'visible', False))

    def _update_lstm_progress(self, progress_data: Dict):
        """Update LSTM training progress."""
        progress_type = progress_data.get('type', '')

        if progress_type == 'lstm_training_start':
            # Show progress panel
            pn.state.execute(lambda: setattr(self.lstm_training_progress, 'visible', True))
            pn.state.execute(lambda: setattr(self.lstm_progress_bar, 'value', 0))

            status_html = f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                <h3 style='margin: 0 0 10px 0;'>🧠 Training LSTM Models for {progress_data.get('symbol', '')}...</h3>
                <p style='margin: 0; opacity: 0.9;'>{progress_data.get('status', '')}</p>
            </div>
            """
            pn.state.execute(lambda: setattr(self.lstm_status_text, 'object', status_html))

        elif progress_type == 'lstm_training_progress':
            # Update progress bar based on epoch and model
            model = progress_data.get('model', 1)
            total_models = progress_data.get('total_models', 3)
            epoch = progress_data.get('epoch', 1)

            # Estimate: assume 50 epochs per model
            estimated_total_epochs = total_models * 50
            current_total_epochs = (model - 1) * 50 + epoch
            progress_percent = int((current_total_epochs / estimated_total_epochs) * 100)

            pn.state.execute(lambda: setattr(self.lstm_progress_bar, 'value', min(progress_percent, 100)))

            # Update status with metrics
            loss = progress_data.get('loss', 0)
            val_loss = progress_data.get('val_loss', 0)
            elapsed = progress_data.get('elapsed_time', 0)

            status_html = f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                <h3 style='margin: 0 0 10px 0;'>🧠 Training LSTM Models for {progress_data.get('symbol', '')}...</h3>
                <p style='margin: 0; opacity: 0.9;'>{progress_data.get('status', '')}</p>
                <div style='margin-top: 10px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
                    <div><small>Loss:</small> <strong>{loss:.4f}</strong></div>
                    <div><small>Val Loss:</small> <strong>{val_loss:.4f}</strong></div>
                    <div><small>Time:</small> <strong>{elapsed:.1f}s</strong></div>
                </div>
            </div>
            """
            pn.state.execute(lambda: setattr(self.lstm_status_text, 'object', status_html))

        elif progress_type == 'lstm_training_complete':
            # Mark model complete
            model = progress_data.get('model', 1)
            total_models = progress_data.get('total_models', 3)

            if model >= total_models:
                # All models complete - hide the progress panel
                # Training history chart will be displayed immediately via callback
                pn.state.execute(lambda: setattr(self.lstm_training_progress, 'visible', False))

    def _display_training_history_immediate(self, training_data: Dict):
        """Display training history immediately after training completes"""
        symbol = training_data.get('symbol', 'Unknown')
        model_info = training_data.get('model_info')

        if not model_info or 'training_histories' not in model_info or not model_info['training_histories']:
            return

        try:
            training_chart = self.visualizer.create_lstm_training_chart(
                model_info['training_histories'],
                symbol
            )

            # Add header for training chart
            training_header = f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 12px; border-radius: 8px; margin: 10px 0;'>
                <h3 style='margin: 0;'>📊 LSTM Training History</h3>
                <p style='margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9;'>
                    Trained: {model_info.get('training_date', 'Unknown')[:10]} |
                    Models: {model_info.get('ensemble_size', 'N/A')} |
                    Format: {model_info.get('model_format', 'N/A')}
                </p>
            </div>
            """

            # Use pn.state.execute to update UI from background thread
            def update_ui():
                self.results_column.append(pn.pane.HTML(training_header))
                self.results_column.append(pn.pane.Plotly(training_chart, sizing_mode="stretch_width", height=600))

            pn.state.execute(update_ui)
            logger.info(f"Training history displayed for {symbol}")

        except Exception as e:
            logger.error(f"Failed to display training history: {e}")

    def _handle_query(self, event=None):
        """Handle query submission with async processing"""
        query = self.query_input.value.strip()

        if not query:
            pn.state.notifications.error("Please enter a query", duration=3000)
            return

        # Clear previous results and hide progress
        self.results_column.clear()
        self.lstm_training_progress.visible = False
        self.lstm_prediction_progress.visible = False

        # Show loading state
        self._set_loading_state(True)

        # Get force retrain checkbox value
        force_retrain = self.force_retrain_checkbox.value

        # Create and run async task
        def run_async_query():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.query_processor.process_query(
                    query,
                    force_retrain=force_retrain,
                    progress_callback=self._update_lstm_progress,
                    training_complete_callback=self._display_training_history_immediate,
                    prediction_callback=self._update_prediction_progress
                ))
                loop.close()

                pn.state.execute(lambda: self._handle_query_result(result, query))

            except Exception as e:
                logger.error(f"Async query processing failed: {e}")
                pn.state.execute(lambda: self._show_error(f"Analysis failed: {str(e)}"))

        import threading
        thread = threading.Thread(target=run_async_query)
        thread.daemon = True
        thread.start()

    def _handle_query_result(self, result: Dict[str, Any], query: str):
        """Handle query result and update UI"""
        logger.info(f"Results ready for: '{query}'")
        try:
            self._display_results(result)

            if result.get('type') == 'error':
                pn.state.notifications.error(str(result.get('message', 'Error')), duration=5000)
            else:
                pn.state.notifications.success(f"Analysis completed: {query}", duration=3000)

        except Exception as e:
            logger.error(f"Result handling failed: {e}")
            self._show_error(f"Failed to display results: {str(e)}")

        finally:
            self._set_loading_state(False)

    def _set_loading_state(self, loading: bool):
        """Set loading state for UI components"""
        self.loading_indicator.value = loading
        self.submit_button.disabled = loading
        self.submit_button.name = "Processing..." if loading else "Analyze"

    def _display_results(self, result: Dict[str, Any]):
        """Display analysis results based on result type"""
        self.results_column.clear()

        result_type = result.get('type', 'unknown')

        if result_type == 'stock_analysis':
            self._display_stock_analysis(result)
        elif result_type == 'prediction':
            self._display_prediction_results(result)
        elif result_type == 'comparison':
            self._display_comparison_results(result)
        elif result_type == 'price_info':
            self._display_price_info(result)
        elif result_type == 'error':
            self._display_error_result(result)
        else:
            self._display_general_response(result)

    def _display_stock_analysis(self, result: Dict[str, Any]):
        """Display comprehensive stock analysis results in a two-column layout."""
        from src.ui.design_system import Colors

        symbol = result.get('symbol', 'Unknown')
        stock_info = result.get('stock_info', {})
        current_data = result.get('current_data', {})

        # Header
        price = current_data.get('current_price', 0)
        prev_close = current_data.get('previous_close', 0)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        change_color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
        change_symbol = '▲' if change >= 0 else '▼'

        # Helper function to format values safely
        def format_value(value, prefix="", suffix="", decimals=2, is_number=False):
            if value is None or (isinstance(value, (int, float)) and value == 0 and not is_number):
                return "N/A"
            if is_number:
                return f"{prefix}{value:,.{decimals}f}{suffix}" if isinstance(value, float) else f"{prefix}{value:,}{suffix}"
            return f"{prefix}{value}{suffix}"

        market_cap = current_data.get('market_cap')
        stats = {
            "Market Cap": format_value(market_cap / 1e9 if market_cap else None, "$", "B", 2, True),
            "P/E Ratio": format_value(stock_info.get('trailingPE'), "", "", 2, True),
            "EPS": format_value(stock_info.get('trailingEps'), "$", "", 2, True),
            "52-Wk High": format_value(stock_info.get('fiftyTwoWeekHigh'), "$", "", 2, True),
            "52-Wk Low": format_value(stock_info.get('fiftyTwoWeekLow'), "$", "", 2, True),
            "Avg Volume": format_value(stock_info.get('averageVolume'), "", "", 0, True)
        }

        # Reduce numeric font further and mute color for better hierarchy
        stats_html = "".join([
            f"<div style='text-align: right;'><div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>{k}</div>"
            f"<div style='font-size: 0.8rem; font-weight: 400; font-family: monospace; color: {Colors.TEXT_SECONDARY};'>{v}</div></div>"
            for k, v in stats.items()
        ])

        header_html = f"""
        <div style='background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-left: 4px solid {Colors.ACCENT_PURPLE}; padding: 12px 15px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='flex: 1; min-width: 140px;'>
                    <h2 style='margin: 0; font-size: 1.5rem; color: {Colors.TEXT_PRIMARY};'>{html.escape(symbol)}</h2>
                    <p style='margin: 3px 0 0 0; color: {Colors.TEXT_SECONDARY}; font-size: 0.8rem;'>{html.escape(stock_info.get('name', symbol))}</p>
                </div>
                <div style='flex: 1; text-align: right; min-width: 180px;'>
                    <div style='font-size: 1.75rem; font-weight: bold; color: {Colors.TEXT_PRIMARY}; font-family: monospace;'>${price:.2f}</div>
                    <div style='font-size: 0.85rem; color: {change_color}; font-weight: 600;'>{change_symbol} {change_pct:+.2f}%</div>
                </div>
                <div style='flex: 4; display: flex; justify-content: space-around; align-items: center; margin-left: 20px; min-width: 640px; gap: 18px;'>
                    {stats_html}
                </div>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(header_html))

        # --- Two-Column Layout ---
        left_column = pn.Column(sizing_mode="stretch_width", scroll=True, max_height=600)
        right_column = pn.Column(sizing_mode="stretch_width", scroll=True, max_height=600) # Scrollable right column

        # --- Left Column (70%) ---
        # Main Chart
        if 'chart_data' in result:
            try:
                chart = self.visualizer.create_price_chart(
                    result['chart_data'], symbol,
                    predictions=result.get('predictions'),
                    technical_indicators=result.get('technical_analysis', {}).get('indicators', {})
                )
                left_column.append(pn.pane.Plotly(chart, sizing_mode="stretch_width", height=500))
            except Exception as e:
                logger.error(f"Chart creation failed: {e}")

        # --- Right Column (30%) ---
        # Trading Signals - stack vertically with full width
        if 'trading_signals' in result:
            right_column.append(self._create_compact_signals_card(result['trading_signals']))

        # Predictions - stack vertically with full width
        if result.get('predictions'):
            right_column.append(self._create_compact_predictions_card(result['predictions']))

        # AI Analysis
        if 'analysis' in result:
            right_column.append(self._create_ai_analysis_card(result['analysis']))

        # Add columns to the main layout
        self.results_column.append(pn.Row(
            pn.Column(left_column, width_policy='max', width=700), # 70%
            pn.Column(right_column, width_policy='max', width=300), # 30%
            sizing_mode="stretch_width"
        ))

    def _create_ai_analysis_card(self, analysis_text: str) -> pn.pane.HTML:
        from src.ui.design_system import Colors
        analysis_html = html.escape(analysis_text).replace('\n', '<br>')
        return pn.pane.HTML(f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid {Colors.BORDER_SUBTLE};'>
                <div style='font-weight: 600; font-size: 0.95rem; color: {Colors.TEXT_PRIMARY}; margin-bottom: 10px;'>🤖 AI Analysis</div>
                <div style='line-height: 1.6; font-size: 0.875rem; color: {Colors.TEXT_SECONDARY};'>{analysis_html}</div>
            </div>
        """)
        # The following line seems to be a duplicate or misplaced return statement. Removing it.
        # return pn.pane.HTML(html_content)

    def _create_technical_analysis_tabs(self, tech_analysis: Dict) -> pn.Tabs:
        tabs = []
        indicators_dict = tech_analysis.get('indicators', {})
        
        try:
            # Convert to DataFrame for better readability
            df = pd.DataFrame(indicators_dict)
            # Format numbers to 4 decimal places for consistency
            df = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).round(4)

            indicators_tab = pn.pane.DataFrame(df, name="Indicators", sizing_mode="stretch_width", max_height=400)
        except Exception as e:
            logger.error(f"Failed to create indicators table: {e}")
            indicators_tab = pn.pane.Alert(f"Could not display indicators: {e}", alert_type='danger')

        tabs.append(indicators_tab)
        return pn.Tabs(*tabs, sizing_mode="stretch_width")

    def _display_prediction_results(self, result: Dict[str, Any]):
        """Display prediction results - compact"""
        from src.ui.design_system import Colors, HTMLComponents

        symbol = result.get('symbol', 'Unknown')
        predictions = result.get('predictions', {})

        # Calculate prediction metrics
        pred_values = predictions.get('predictions', [])
        last_price = predictions.get('last_price', 0)

        if pred_values:
            avg_pred = sum(pred_values) / len(pred_values)
            change_pct = ((avg_pred - last_price) / last_price * 100) if last_price else 0
            color = Colors.SUCCESS_GREEN if change_pct >= 0 else Colors.DANGER_RED
            symbol = '▲' if change_pct >= 0 else '▼'
        else:
            avg_pred = last_price
            change_pct = 0
            color = Colors.TEXT_MUTED
            symbol = '→'

        # Training history is only shown when training actually happens
        # It's displayed immediately via training_complete_callback
        # No need to show it here for existing models

        # Display prediction header and chart
        header_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-left: 44px solid {Colors.INFO_BLUE};
                    padding: 12px;
                    border-radius: 8px;
                    margin: 20px 0 10px 0;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h3 style='margin: 0; color: {Colors.TEXT_PRIMARY};'>🔮 Price Prediction: {html.escape(symbol)}</h3>
            <p style='margin: 5px 0 0 0; font-size: 0.9em; color: {Colors.TEXT_SECONDARY};'>
                30-Day: ${avg_pred:.2f} |
                Change: <span style='color: {color}; font-weight: 600;'>{symbol} {abs(change_pct):.1f}%</span> |
                Current: ${last_price:.2f}
            </p>
        </div>
        """
        self.results_column.append(pn.pane.HTML(header_html))
        if 'chart_data' in result and predictions:
            try:
                chart = self.visualizer.create_prediction_chart(
                    result['chart_data'],
                    predictions,
                    symbol
                )
                self.results_column.append(pn.pane.Plotly(chart, sizing_mode="stretch_width", height=400))
            except Exception as e:
                logger.error(f"Prediction chart failed: {e}")

    def _display_comparison_results(self, result: Dict[str, Any]):
        """Display stock comparison"""
        from src.ui.design_system import Colors, HTMLComponents

        symbols = result.get('symbols', [])
        comparison_data = result.get('comparison_data', {})

        header_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-left: 4px solid {Colors.ACCENT_CYAN};
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h2 style='margin: 0; color: {Colors.TEXT_PRIMARY};'>⚖️ Comparison: {' vs '.join([html.escape(s) for s in symbols])}</h2>
        </div>
        """
        self.results_column.append(pn.pane.HTML(header_html))

        if comparison_data:
            try:
                chart = self.visualizer.create_comparison_chart(comparison_data)
                self.results_column.append(pn.pane.Plotly(chart, sizing_mode="stretch_width", height=400))
            except Exception as e:
                logger.error(f"Comparison chart failed: {e}")

    def _display_price_info(self, result: Dict[str, Any]):
        """Display current price"""
        from src.ui.design_system import Colors, HTMLComponents

        symbol = result.get('symbol', 'Unknown')
        current_data = result.get('current_data', {})
        stock_info = result.get('stock_info', {})

        price = current_data.get('current_price', 0)
        prev_close = current_data.get('previous_close', 0)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        price_color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
        price_html = f"""
        <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; border-left: 4px solid {Colors.ACCENT_PURPLE};'>
            <h2 style='margin: 0 0 10px 0; color: {Colors.TEXT_PRIMARY};'>💰 {html.escape(symbol)} - {html.escape(stock_info.get('name', ''))}</h2>
            <div style='font-size: 32px; font-weight: bold; margin: 10px 0; color: {Colors.TEXT_PRIMARY};'>${price:.2f}</div>
            <div style='color: {price_color}; font-size: 18px;'>
                {"▲" if change >= 0 else "▼"} ${abs(change):.2f} ({change_pct:+.2f}%)
            </div>
            <div style='margin-top: 15px; color: {Colors.TEXT_SECONDARY};
                <div>Previous Close: ${prev_close:.2f}</div>
                <div>Volume: {current_data.get('volume', 0):,}</div>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(price_html))

    def _display_error_result(self, result: Dict[str, Any]):
        """Display error"""
        from src.ui.design_system import Colors, HTMLComponents

        error_html = f"""
        <div style='background: {Colors.DANGER_BG}; padding: 20px; border-radius: 8px; border-left: 4px solid {Colors.DANGER_RED};
            <h3 style='margin: 0 0 10px 0; color: {Colors.DANGER_RED};'>❌ Error</h3>
            <p style='color: {Colors.TEXT_PRIMARY};'>{html.escape(str(result.get('message', 'Unknown error')))}</p>
            <div style='margin-top: 15px; font-size: 14px; color: {Colors.TEXT_SECONDARY};
                <strong>Try:</strong> "Analyze AAPL" • "Predict GOOGL" • "Compare MSFT vs AAPL"
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(error_html))

    def _display_general_response(self, result: Dict[str, Any]):
        """Display general response"""
        from src.ui.design_system import Colors, HTMLComponents

        message = html.escape(str(result.get('message', 'How can I help?')))
        response_html = f"""
        <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
            <h3 style='margin: 0 0 10px 0; color: {Colors.TEXT_PRIMARY};'>💡 {message}</h3>
            <p style='color: {Colors.TEXT_SECONDARY};'>Try: "Analyze AAPL" • "Predict GOOGL" • "Compare stocks"</p>
        </div>
        """
        self.results_column.append(pn.pane.HTML(response_html))

    def _create_compact_signals_card(self, trading_signals: Dict[str, Any]) -> pn.pane.HTML:
        """Create compact trading signals card"""
        from src.ui.design_system import Colors, HTMLComponents

        signal = trading_signals.get('primary_signal', 'HOLD')
        confidence = trading_signals.get('confidence', 0)
        recommendations = trading_signals.get('recommendations', [])[:2]

        signal_colors = {'BUY': Colors.SUCCESS_GREEN, 'SELL': Colors.DANGER_RED, 'HOLD': Colors.WARNING_YELLOW}
        color = signal_colors.get(signal, Colors.TEXT_MUTED)

        recs_html = ''.join([f"<li style='font-size: 13px;'>{html.escape(str(r))}</li>" for r in recommendations])

        html_content = f"""
        <div style='background: {Colors.BG_PRIMARY}; padding: 15px; border-radius: 8px; border: 1px solid {Colors.BORDER_SUBTLE}; margin-bottom: 10px; width: 100%;'>
            <div style='font-size: 13px; color: {Colors.TEXT_SECONDARY}; margin-bottom: 5px;'>Trading Signal</div>
            <div style='font-size: 24px; font-weight: bold; color: {color};'>{signal}</div>
            <div style='font-size: 12px; color: {Colors.TEXT_MUTED}; margin: 5px 0;'>{confidence:.0f}% confidence</div>
            <ul style='margin: 10px 0 0 0; padding-left: 20px; color: {Colors.TEXT_SECONDARY};'>{recs_html}</ul>
        </div>
        """
        return pn.pane.HTML(html_content)

    def _create_compact_predictions_card(self, predictions: Dict[str, Any]) -> pn.pane.HTML:
        """Create compact predictions card"""
        from src.ui.design_system import Colors, HTMLComponents

        pred_values = predictions.get('predictions', [])
        last_price = predictions.get('last_price', 0)

        if pred_values:
            avg_pred = sum(pred_values) / len(pred_values)
            change_pct = ((avg_pred - last_price) / last_price * 100) if last_price else 0
            color = Colors.SUCCESS_GREEN if change_pct >= 0 else Colors.DANGER_RED
            symbol = '▲' if change_pct >= 0 else '▼'
        else:
            avg_pred = last_price
            change_pct = 0
            color = Colors.TEXT_MUTED
            symbol = '→'

        html_content = f"""
        <div style='background: {Colors.BG_PRIMARY}; padding: 15px; border-radius: 8px; border: 1px solid {Colors.BORDER_SUBTLE}; margin-bottom: 10px; width: 100%;'>
            <div style='font-size: 13px; color: {Colors.TEXT_SECONDARY}; margin-bottom: 5px;'>30-Day Prediction</div>
            <div style='font-size: 24px; font-weight: bold; color: {Colors.TEXT_PRIMARY};'>${avg_pred:.2f}</div>
            <div style='font-size: 14px; color: {color}; margin: 5px 0;'>
                {symbol} {change_pct:+.1f}%
            </div>
            <div style='font-size: 12px; color: {Colors.TEXT_MUTED};'>
                Current: ${last_price:.2f}
            </div>
            <div style='font-size: 11px; color: {Colors.TEXT_MUTED}; margin-top: 8px; font-style: italic;'>
                ⚠️ Not financial advice
            </div>
        </div>
        """
        return pn.pane.HTML(html_content)



    def _show_error(self, message: str):
        """Show error notification"""
        pn.state.notifications.error(html.escape(str(message)), duration=5000)
        self._set_loading_state(False)

    def get_analysis_tab(self):
        """Get the analysis tab layout"""
        # Input controls section with similar style to RL Trading
        from src.ui.design_system import Colors

        input_section = pn.Column(
            pn.Row(
                self.query_input,
                self.submit_button,
                self.loading_indicator,
                sizing_mode="stretch_width"
            ),
            self.force_retrain_checkbox,
            self.quick_buttons,
            styles=dict(background=Colors.BG_SECONDARY, border_radius='8px', padding='15px'),
            margin=(0, 0, 15, 0)
        )

        # LSTM Training progress section
        self.lstm_training_progress.clear()
        self.lstm_training_progress.extend([
            self.lstm_status_text,
            self.lstm_progress_bar
        ])

        # LSTM Prediction progress section
        self.lstm_prediction_progress.clear()
        self.lstm_prediction_progress.extend([
            self.lstm_prediction_text,
            self.lstm_prediction_bar
        ])

        # Disclaimer at bottom
        from src.ui.design_system import HTMLComponents
        disclaimer_html = HTMLComponents.disclaimer()

        return pn.Column(
            input_section,
            self.lstm_training_progress,
            self.lstm_prediction_progress,
            self.results_column,
            pn.pane.HTML(disclaimer_html),
            sizing_mode="stretch_width",
        )
