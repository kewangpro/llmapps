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
        symbol = result.get('symbol', 'Unknown')
        stock_info = result.get('stock_info', {})
        current_data = result.get('current_data', {})

        # Compact header with price
        price = current_data.get('current_price', 0)
        prev_close = current_data.get('previous_close', 0)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        change_color = '#10b981' if change >= 0 else '#ef4444'
        change_symbol = '▲' if change >= 0 else '▼'

        header_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 style='margin: 0; font-size: 24px;'>{html.escape(symbol)}</h2>
                    <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 14px;'>{html.escape(stock_info.get('name', symbol))}</p>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 28px; font-weight: bold;'>${price:.2f}</div>
                    <div style='font-size: 14px; color: {change_color};'>
                        {change_symbol} ${abs(change):.2f} ({change_pct:+.2f}%)
                    </div>
                </div>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(header_html))

        # Main chart
        if 'chart_data' in result:
            try:
                chart = self.visualizer.create_price_chart(
                    result['chart_data'],
                    symbol,
                    predictions=result.get('predictions'),
                    technical_indicators=result.get('technical_analysis', {}).get('indicators', {})
                )
                self.results_column.append(pn.pane.Plotly(chart, sizing_mode="stretch_width", height=400))
            except Exception as e:
                logger.error(f"Chart creation failed: {e}")

        # Compact cards row
        cards_row = pn.Row(sizing_mode="stretch_width")

        # Trading signals card (compact)
        if 'trading_signals' in result:
            cards_row.append(self._create_compact_signals_card(result['trading_signals']))

        # Predictions card (compact)
        if result.get('predictions'):
            cards_row.append(self._create_compact_predictions_card(result['predictions']))

        if len(cards_row) > 0:
            self.results_column.append(cards_row)

        # AI Analysis (collapsible)
        if 'analysis' in result:
            analysis_text = html.escape(str(result['analysis'])).replace('\n', '<br>')
            analysis_html = f"""
            <details style='background: #f3f4f6; padding: 15px; border-radius: 8px; margin-top: 10px;'>
                <summary style='cursor: pointer; font-weight: bold; font-size: 16px;'>🤖 AI Analysis</summary>
                <div style='margin-top: 10px; line-height: 1.6;'>{analysis_text}</div>
            </details>
            """
            self.results_column.append(pn.pane.HTML(analysis_html))

    def _display_prediction_results(self, result: Dict[str, Any]):
        """Display prediction results - compact"""
        symbol = result.get('symbol', 'Unknown')
        predictions = result.get('predictions', {})

        # Calculate prediction metrics
        pred_values = predictions.get('predictions', [])
        last_price = predictions.get('last_price', 0)

        if pred_values:
            avg_pred = sum(pred_values) / len(pred_values)
            change_pct = ((avg_pred - last_price) / last_price * 100) if last_price else 0
            change_color = '#10b981' if change_pct >= 0 else '#ef4444'
            change_symbol = '▲' if change_pct >= 0 else '▼'
        else:
            avg_pred = last_price
            change_pct = 0
            change_color = '#9ca3af'
            change_symbol = '→'

        # Training history is only shown when training actually happens
        # It's displayed immediately via training_complete_callback
        # No need to show it here for existing models

        # Display prediction header and chart
        header_html = f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white; padding: 12px; border-radius: 8px; margin: 20px 0 10px 0;'>
            <h3 style='margin: 0;'>🔮 Price Prediction: {html.escape(symbol)}</h3>
            <p style='margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9;'>
                30-Day: ${avg_pred:.2f} |
                Change: <span style='color: {change_color};'>{change_symbol} {abs(change_pct):.1f}%</span> |
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
        symbols = result.get('symbols', [])
        comparison_data = result.get('comparison_data', {})

        header_html = f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
            <h2 style='margin: 0;'>⚖️ Comparison: {' vs '.join([html.escape(s) for s in symbols])}</h2>
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
        symbol = result.get('symbol', 'Unknown')
        current_data = result.get('current_data', {})
        stock_info = result.get('stock_info', {})

        price = current_data.get('current_price', 0)
        prev_close = current_data.get('previous_close', 0)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        price_html = f"""
        <div style='background: #f9fafb; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea;'>
            <h2 style='margin: 0 0 10px 0;'>💰 {html.escape(symbol)} - {html.escape(stock_info.get('name', ''))}</h2>
            <div style='font-size: 32px; font-weight: bold; margin: 10px 0;'>${price:.2f}</div>
            <div style='color: {"#10b981" if change >= 0 else "#ef4444"}; font-size: 18px;'>
                {"▲" if change >= 0 else "▼"} ${abs(change):.2f} ({change_pct:+.2f}%)
            </div>
            <div style='margin-top: 15px; color: #6b7280;'>
                <div>Previous Close: ${prev_close:.2f}</div>
                <div>Volume: {current_data.get('volume', 0):,}</div>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(price_html))

    def _display_error_result(self, result: Dict[str, Any]):
        """Display error"""
        error_html = f"""
        <div style='background: #fee2e2; padding: 20px; border-radius: 8px; border-left: 4px solid #ef4444;'>
            <h3 style='margin: 0 0 10px 0; color: #991b1b;'>❌ Error</h3>
            <p style='color: #7f1d1d;'>{html.escape(str(result.get('message', 'Unknown error')))}</p>
            <div style='margin-top: 15px; font-size: 14px; color: #6b7280;'>
                <strong>Try:</strong> "Analyze AAPL" • "Predict GOOGL" • "Compare MSFT vs AAPL"
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(error_html))

    def _display_general_response(self, result: Dict[str, Any]):
        """Display general response"""
        message = html.escape(str(result.get('message', 'How can I help?')))
        response_html = f"""
        <div style='background: #f3f4f6; padding: 20px; border-radius: 8px;'>
            <h3 style='margin: 0 0 10px 0;'>💡 {message}</h3>
            <p style='color: #6b7280;'>Try: "Analyze AAPL" • "Predict GOOGL" • "Compare stocks"</p>
        </div>
        """
        self.results_column.append(pn.pane.HTML(response_html))

    def _create_compact_signals_card(self, trading_signals: Dict[str, Any]) -> pn.pane.HTML:
        """Create compact trading signals card"""
        signal = trading_signals.get('primary_signal', 'HOLD')
        confidence = trading_signals.get('confidence', 0)
        recommendations = trading_signals.get('recommendations', [])[:2]

        signal_colors = {'BUY': '#10b981', 'SELL': '#ef4444', 'HOLD': '#f59e0b'}
        color = signal_colors.get(signal, '#6b7280')

        recs_html = ''.join([f"<li style='font-size: 13px;'>{html.escape(str(r))}</li>" for r in recommendations])

        html_content = f"""
        <div style='background: white; padding: 15px; border-radius: 8px; border: 1px solid #e5e7eb; flex: 1;'>
            <div style='font-size: 13px; color: #6b7280; margin-bottom: 5px;'>Trading Signal</div>
            <div style='font-size: 24px; font-weight: bold; color: {color};'>{signal}</div>
            <div style='font-size: 12px; color: #9ca3af; margin: 5px 0;'>{confidence:.0f}% confidence</div>
            <ul style='margin: 10px 0 0 0; padding-left: 20px; color: #4b5563;'>{recs_html}</ul>
        </div>
        """
        return pn.pane.HTML(html_content)

    def _create_compact_predictions_card(self, predictions: Dict[str, Any]) -> pn.pane.HTML:
        """Create compact predictions card"""
        pred_values = predictions.get('predictions', [])
        last_price = predictions.get('last_price', 0)

        if pred_values:
            avg_pred = sum(pred_values) / len(pred_values)
            change_pct = ((avg_pred - last_price) / last_price * 100) if last_price else 0
            color = '#10b981' if change_pct >= 0 else '#ef4444'
            symbol = '▲' if change_pct >= 0 else '▼'
        else:
            avg_pred = last_price
            change_pct = 0
            color = '#6b7280'
            symbol = '→'

        html_content = f"""
        <div style='background: white; padding: 15px; border-radius: 8px; border: 1px solid #e5e7eb; flex: 1;'>
            <div style='font-size: 13px; color: #6b7280; margin-bottom: 5px;'>30-Day Prediction</div>
            <div style='font-size: 24px; font-weight: bold;'>${avg_pred:.2f}</div>
            <div style='font-size: 14px; color: {color}; margin: 5px 0;'>
                {symbol} {change_pct:+.1f}%
            </div>
            <div style='font-size: 12px; color: #9ca3af;'>
                Current: ${last_price:.2f}
            </div>
            <div style='font-size: 11px; color: #9ca3af; margin-top: 8px; font-style: italic;'>
                ⚠️ Not financial advice
            </div>
        </div>
        """
        return pn.pane.HTML(html_content)

    def _show_welcome(self):
        """Show compact welcome message"""
        welcome_html = """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 25px; border-radius: 10px; text-align: center;'>
            <h2 style='margin: 0 0 10px 0;'>📈 Stock Analysis and Trading AI</h2>
            <p style='margin: 0 0 20px 0; opacity: 0.9;'>AI-powered stock analysis, predictions, and RL trading strategies</p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; text-align: left;'>
                <div style='background: rgba(255,255,255,0.1); padding: 12px; border-radius: 6px;'>
                    <div style='font-weight: bold; margin-bottom: 5px;'>📊 Analysis</div>
                    <div style='font-size: 13px; opacity: 0.9;'>Technical analysis & signals</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 12px; border-radius: 6px;'>
                    <div style='font-weight: bold; margin-bottom: 5px;'>🔮 Predictions</div>
                    <div style='font-size: 13px; opacity: 0.9;'>LSTM price forecasting</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 12px; border-radius: 6px;'>
                    <div style='font-weight: bold; margin-bottom: 5px;'>🤖 RL Trading</div>
                    <div style='font-size: 13px; opacity: 0.9;'>Train AI trading agents</div>
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
        input_section = pn.Column(
            pn.Row(
                self.query_input,
                self.submit_button,
                self.loading_indicator,
                sizing_mode="stretch_width"
            ),
            self.force_retrain_checkbox,
            self.quick_buttons,
            styles=dict(background='#f9fafb', border_radius='8px', padding='15px'),
            margin=(0, 0, 15, 0)
        )

        # LSTM Training progress section
        self.lstm_training_progress.clear()
        self.lstm_training_progress.extend([
            self.lstm_status_text,
            self.lstm_progress_bar
        ])

        # Disclaimer at bottom
        disclaimer_html = """
        <div style='background: #fef3c7; border: 1px solid #fbbf24; padding: 15px; border-radius: 8px; margin-top: 20px;'>
            <h4 style='margin: 0 0 10px 0; color: #92400e;'>⚠️ Educational Disclaimer</h4>
            <ul style='margin: 0; padding-left: 20px; font-size: 12px; color: #78350f; line-height: 1.6;'>
                <li><strong>Educational purpose only.</strong> Not financial advice.</li>
                <li><strong>AI Analysis:</strong> Powered by Ollama and LSTM models for learning purposes</li>
                <li><strong>Technical Indicators:</strong> Past performance doesn't guarantee future results</li>
                <li><strong>Predictions:</strong> 30-day forecasts are experimental and for educational use only</li>
            </ul>
            <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #fbbf24; font-size: 11px; color: #78350f;'>
                Always consult qualified financial professionals before making investment decisions.
            </div>
        </div>
        """

        return pn.Column(
            input_section,
            self.lstm_training_progress,
            self.results_column,
            pn.pane.HTML(disclaimer_html),
            sizing_mode="stretch_width",
        )


def create_app():
    """Create and configure the Panel application with tabs"""
    from src.ui.rl_components import CompactRLPanel

    # Create main app
    main_app = StockAnalysisApp()

    # Create RL panel
    rl_panel = CompactRLPanel()

    # Create tabs
    tabs = pn.Tabs(
        ('📊 Analysis', main_app.get_analysis_tab()),
        ('🤖 RL Trading', rl_panel.get_panel()),
        dynamic=True,
        sizing_mode="stretch_width",
        tabs_location='above',
        active=0
    )

    # Main layout (no header)
    layout = pn.Column(
        tabs,
        sizing_mode="stretch_width",
        max_width=1400,
        margin=(10, 20)
    )

    # Create template
    template = pn.template.FastListTemplate(
        title="Stock Analysis and Trading AI",
        sidebar=[],
        header_background='#1f2937',
        theme='default',
        main_max_width='1400px',
    )
    template.main.append(layout)

    return template.servable()
