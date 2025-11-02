import panel as pn
import param
import asyncio
import html
from typing import Dict, Any, Optional, Tuple
import logging

from src.agents.query_processor import QueryProcessor
from src.tools.visualizer import Visualizer

logger = logging.getLogger(__name__)

# Configure Panel
pn.extension('plotly', template='bootstrap', notifications=True)

class StockAnalysisApp(param.Parameterized):
    """Main Stock Analysis Panel Application with RL Integration"""

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize components
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

        # Quick action buttons - more compact
        quick_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
        self.quick_buttons = pn.Row(
            *[pn.widgets.Button(name=sym, button_type="light", width=70, height=30)
              for sym in quick_stocks],
            sizing_mode="stretch_width",
        )

        # Bind events
        self.submit_button.on_click(self._handle_query)
        self.query_input.param.watch(self._on_enter_key, 'value')

        for button in self.quick_buttons:
            button.on_click(self._handle_quick_button)

        # Initial welcome message
        self._show_welcome()

    def _on_enter_key(self, event):
        """Handle Enter key press"""
        pass  # Panel doesn't have native Enter support

    def _handle_quick_button(self, event):
        """Handle quick action button clicks"""
        symbol = event.obj.name
        self.query_input.value = f"Analyze {symbol}"
        self._handle_query()

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
                    training_complete_callback=self._display_training_history_immediate
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
        """Display comprehensive stock analysis results - compact version"""
        # Import design system
        from src.ui.design_system import Colors, HTMLComponents

        symbol = result.get('symbol', 'Unknown')
        stock_info = result.get('stock_info', {})
        current_data = result.get('current_data', {})

        # Compact header with price
        price = current_data.get('current_price', 0)
        prev_close = current_data.get('previous_close', 0)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        change_color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
        change_symbol = '▲' if change >= 0 else '▼'

        header_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-left: 4px solid {Colors.ACCENT_PURPLE};
                    padding: 12px 15px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 style='margin: 0; font-size: 1.5rem; color: {Colors.TEXT_PRIMARY};'>{html.escape(symbol)}</h2>
                    <p style='margin: 3px 0 0 0; color: {Colors.TEXT_SECONDARY}; font-size: 0.8rem;'>{html.escape(stock_info.get('name', symbol))}</p>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 1.75rem; font-weight: bold; color: {Colors.TEXT_PRIMARY}; font-family: monospace;'>${price:.2f}</div>
                    <div style='font-size: 0.85rem; color: {change_color}; font-weight: 600;'>
                        {change_symbol} {change_pct:+.2f}%
                    </div>
                </div>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(header_html))

        # Create side-by-side layout: Chart (left 60%) + Info cards (right 40%)
        left_column = pn.Column(sizing_mode="stretch_width")
        right_column = pn.Column(sizing_mode="stretch_width", max_width=450)

        # Main chart on left
        if 'chart_data' in result:
            try:
                chart = self.visualizer.create_price_chart(
                    result['chart_data'],
                    symbol,
                    predictions=result.get('predictions'),
                    technical_indicators=result.get('technical_analysis', {}).get('indicators', {})
                )
                left_column.append(pn.pane.Plotly(chart, sizing_mode="stretch_width", height=500))
            except Exception as e:
                logger.error(f"Chart creation failed: {e}")

        # Info cards on right (stacked vertically)
        # Trading signals card (compact)
        if 'trading_signals' in result:
            right_column.append(self._create_compact_signals_card(result['trading_signals']))

        # Predictions card (compact)
        if result.get('predictions'):
            right_column.append(self._create_compact_predictions_card(result['predictions']))

        # Add side-by-side row
        self.results_column.append(
            pn.Row(
                left_column,
                right_column,
                sizing_mode="stretch_width"
            )
        )

        # AI Analysis (collapsible) - full width at bottom
        if 'analysis' in result:
            analysis_text = html.escape(str(result['analysis'])).replace('\n', '<br>')
            analysis_html = f"""
            <details style='background: {Colors.BG_SECONDARY}; padding: 12px; border-radius: 8px; margin-top: 10px; border: 1px solid {Colors.BORDER_SUBTLE};'>
                <summary style='cursor: pointer; font-weight: 600; font-size: 0.95rem; color: {Colors.TEXT_PRIMARY};'>🤖 AI Analysis</summary>
                <div style='margin-top: 10px; line-height: 1.6; font-size: 0.875rem; color: {Colors.TEXT_SECONDARY};'>{analysis_text}</div>
            </details>
            """
            self.results_column.append(pn.pane.HTML(analysis_html))

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
            change_color = Colors.SUCCESS_GREEN if change_pct >= 0 else Colors.DANGER_RED
            change_symbol = '▲' if change_pct >= 0 else '▼'
        else:
            avg_pred = last_price
            change_pct = 0
            change_color = Colors.TEXT_MUTED
            change_symbol = '→'

        # Training history is only shown when training actually happens
        # It's displayed immediately via training_complete_callback
        # No need to show it here for existing models

        # Display prediction header and chart
        header_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-left: 4px solid {Colors.INFO_BLUE};
                    padding: 12px;
                    border-radius: 8px;
                    margin: 20px 0 10px 0;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h3 style='margin: 0; color: {Colors.TEXT_PRIMARY};'>🔮 Price Prediction: {html.escape(symbol)}</h3>
            <p style='margin: 5px 0 0 0; font-size: 0.9em; color: {Colors.TEXT_SECONDARY};'>
                30-Day: ${avg_pred:.2f} |
                Change: <span style='color: {change_color}; font-weight: 600;'>{change_symbol} {abs(change_pct):.1f}%</span> |
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
            <div style='margin-top: 15px; color: {Colors.TEXT_SECONDARY};'>
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
        <div style='background: {Colors.DANGER_BG}; padding: 20px; border-radius: 8px; border-left: 4px solid {Colors.DANGER_RED};'>
            <h3 style='margin: 0 0 10px 0; color: {Colors.DANGER_RED};'>❌ Error</h3>
            <p style='color: {Colors.TEXT_PRIMARY};'>{html.escape(str(result.get('message', 'Unknown error')))}</p>
            <div style='margin-top: 15px; font-size: 14px; color: {Colors.TEXT_SECONDARY};'>
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
        <div style='background: {Colors.BG_PRIMARY}; padding: 15px; border-radius: 8px; border: 1px solid {Colors.BORDER_SUBTLE}; flex: 1;'>
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
        <div style='background: {Colors.BG_PRIMARY}; padding: 15px; border-radius: 8px; border: 1px solid {Colors.BORDER_SUBTLE}; flex: 1;'>
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

    def _show_welcome(self):
        """Show professional welcome message"""
        from src.ui.design_system import Colors

        welcome_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    padding: 25px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <h2 style='margin: 0 0 10px 0; color: {Colors.TEXT_PRIMARY};'>📈 Stock Analysis and Trading AI</h2>
            <p style='margin: 0 0 20px 0; color: {Colors.TEXT_SECONDARY};'>AI-powered stock analysis, predictions, and RL trading strategies</p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; text-align: left;'>
                <div style='background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; padding: 12px; border-radius: 6px;'>
                    <div style='font-weight: bold; margin-bottom: 5px; color: {Colors.TEXT_PRIMARY};'>📊 Analysis</div>
                    <div style='font-size: 13px; color: {Colors.TEXT_SECONDARY};'>Technical analysis & signals</div>
                </div>
                <div style='background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; padding: 12px; border-radius: 6px;'>
                    <div style='font-weight: bold; margin-bottom: 5px; color: {Colors.TEXT_PRIMARY};'>🔮 Predictions</div>
                    <div style='font-size: 13px; color: {Colors.TEXT_SECONDARY};'>LSTM price forecasting</div>
                </div>
                <div style='background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; padding: 12px; border-radius: 6px;'>
                    <div style='font-weight: bold; margin-bottom: 5px; color: {Colors.TEXT_PRIMARY};'>🤖 RL Trading</div>
                    <div style='font-size: 13px; color: {Colors.TEXT_SECONDARY};'>Train AI trading agents</div>
                </div>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(welcome_html))

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

        # Disclaimer at bottom
        from src.ui.design_system import HTMLComponents
        disclaimer_html = HTMLComponents.disclaimer()

        return pn.Column(
            input_section,
            self.lstm_training_progress,
            self.results_column,
            pn.pane.HTML(disclaimer_html),
            sizing_mode="stretch_width",
        )


def create_app():
    """Create and configure the Panel application with professional navigation"""
    from src.ui.rl_components import CompactRLPanel
    from src.ui.pages.dashboard import DashboardPage
    from src.ui.pages.portfolio import PortfolioPage
    from src.ui.pages.models import ModelsPage
    from src.ui.design_system import Colors

    # Create all pages
    dashboard_page = DashboardPage()
    analysis_app = StockAnalysisApp()
    rl_panel = CompactRLPanel()
    portfolio_page = PortfolioPage()
    models_page = ModelsPage()

    # Create professional navigation tabs
    tabs = pn.Tabs(
        ('📊 Dashboard', dashboard_page.get_view()),
        ('📈 Analysis', analysis_app.get_analysis_tab()),
        ('🤖 Trading', rl_panel.get_panel()),
        ('💼 Portfolio', portfolio_page.get_view()),
        ('🧠 Models', models_page.get_view()),
        dynamic=True,
        sizing_mode="stretch_width",
        tabs_location='above',
        active=0
    )

    # Main layout with professional styling
    layout = pn.Column(
        tabs,
        sizing_mode="stretch_width",
        max_width=1600,
        margin=(0, 0)
    )

    # Create template with light theme
    template = pn.template.FastListTemplate(
        title="Stock Agent Pro",
        sidebar=[],
        header_background=Colors.ACCENT_PURPLE,  # Use purple for better visibility
        theme='default',
        main_max_width='1600px',
        theme_toggle=False,  # Disable theme toggle
    )
    template.main.append(layout)

    return template.servable()
