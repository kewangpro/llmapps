import panel as pn
import param
import asyncio
from typing import Dict, Any, Optional
import logging

from src.agents.query_processor import QueryProcessor
from src.tools.visualizer import Visualizer

logger = logging.getLogger(__name__)

# Configure Panel
pn.extension('plotly', template='bootstrap')

class StockAnalysisApp(param.Parameterized):
    """Main Stock Analysis Panel Application"""
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.visualizer = Visualizer()
        
        # UI Components
        self.query_input = pn.widgets.TextInput(
            placeholder="Ask about stocks... (e.g., 'Analyze AAPL for the last 6 months')",
            sizing_mode="stretch_width",
            height=50,
            css_classes=['query-input']
        )
        
        self.submit_button = pn.widgets.Button(
            name="🔍 Analyze", 
            button_type="primary",
            width=120,
            css_classes=['submit-button']
        )
        
        self.loading_indicator = pn.indicators.LoadingSpinner(
            value=False, 
            size=30, 
            color='primary'
        )
        
        self.results_column = pn.Column(
            sizing_mode="stretch_width",
            min_height=400,
            css_classes=['results-container']
        )
        
        self.status_bar = pn.pane.HTML(
            "<div class='status-ready'>💡 Ready to analyze stocks...</div>",
            sizing_mode="stretch_width",
            css_classes=['status-bar']
        )
        
        # Quick action buttons
        self.quick_buttons = pn.Row(
            pn.widgets.Button(name="📈 AAPL", button_type="light", width=80),
            pn.widgets.Button(name="🔍 GOOGL", button_type="light", width=80),
            pn.widgets.Button(name="⚡ TSLA", button_type="light", width=80),
            pn.widgets.Button(name="🏢 MSFT", button_type="light", width=80),
            pn.widgets.Button(name="🛒 AMZN", button_type="light", width=80),
            sizing_mode="stretch_width",
            css_classes=['quick-buttons']
        )
        
        # Bind events
        self.submit_button.on_click(self._handle_query)
        self.query_input.param.watch(self._on_input_change, 'value')
        
        # Bind quick button events
        for button in self.quick_buttons:
            button.on_click(self._handle_quick_button)
        
        # Initial welcome message
        self._show_welcome()
    
    def _on_input_change(self, event):
        """Handle input change for Enter key simulation"""
        # This is a simple approach - in a real app you might use JavaScript
        pass
    
    def _handle_quick_button(self, event):
        """Handle quick action button clicks"""
        button_name = event.obj.name
        symbol = button_name.split()[-1]  # Extract symbol from button name
        self.query_input.value = f"Analyze {symbol}"
        self._handle_query()
    
    def _handle_query(self, event=None):
        """Handle query submission with async processing"""
        query = self.query_input.value.strip()
        
        if not query:
            self._show_error("Please enter a query")
            return
        
        # Show loading state
        self._set_loading_state(True)
        
        # Create and run async task
        def run_async_query():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.query_processor.process_query(query))
                loop.close()
                
                # Schedule UI update on main thread
                pn.state.execute(lambda: self._handle_query_result(result, query))
                
            except Exception as e:
                logger.error(f"Async query processing failed: {e}")
                pn.state.execute(lambda: self._show_error(f"Analysis failed: {str(e)}"))
        
        # Run in thread to avoid blocking UI
        import threading
        thread = threading.Thread(target=run_async_query)
        thread.daemon = True
        thread.start()
    
    def _handle_query_result(self, result: Dict[str, Any], query: str):
        """Handle query result and update UI"""
        try:
            # Display results
            self._display_results(result)
            
            # Update status
            if result.get('type') == 'error':
                self.status_bar.object = f"<div class='status-error'>❌ {result.get('message', 'Error occurred')}</div>"
            else:
                self.status_bar.object = f"<div class='status-success'>✅ Analysis completed: {query}</div>"
                
        except Exception as e:
            logger.error(f"Result handling failed: {e}")
            self._show_error(f"Failed to display results: {str(e)}")
            
        finally:
            # Reset loading state
            self._set_loading_state(False)
    
    def _set_loading_state(self, loading: bool):
        """Set loading state for UI components"""
        self.loading_indicator.value = loading
        self.submit_button.disabled = loading
        
        if loading:
            self.status_bar.object = "<div class='status-loading'>🔄 Processing your query...</div>"
    
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
        """Display comprehensive stock analysis results"""
        symbol = result.get('symbol', 'Unknown')
        stock_info = result.get('stock_info', {})
        current_data = result.get('current_data', {})
        
        # Title and basic info
        title_html = f"""
        <div class='analysis-header'>
            <h2>📊 Stock Analysis: {symbol}</h2>
            <h4>{stock_info.get('name', symbol)}</h4>
        </div>
        """
        self.results_column.append(pn.pane.HTML(title_html))
        
        # Current price card
        if current_data:
            price_card = self._create_price_card(current_data, stock_info)
            self.results_column.append(price_card)
        
        # Main price chart
        if 'chart_data' in result:
            try:
                chart = self.visualizer.create_price_chart(
                    result['chart_data'], 
                    symbol, 
                    predictions=result.get('predictions'),
                    technical_indicators=result.get('technical_analysis', {}).get('indicators', {})
                )
                chart_pane = pn.pane.Plotly(chart, sizing_mode="stretch_width", height=500)
                self.results_column.append(chart_pane)
            except Exception as e:
                logger.error(f"Chart creation failed: {e}")
                self.results_column.append(pn.pane.HTML("<div class='error'>Failed to create chart</div>"))
        
        # Technical analysis summary
        if 'trading_signals' in result:
            signals_card = self._create_signals_card(result['trading_signals'])
            self.results_column.append(signals_card)
        
        # Predictions if available
        if result.get('predictions'):
            predictions_card = self._create_predictions_card(result['predictions'])
            self.results_column.append(predictions_card)
        
        # AI Analysis text
        if 'analysis' in result:
            analysis_card = self._create_analysis_card(result['analysis'])
            self.results_column.append(analysis_card)
    
    def _display_prediction_results(self, result: Dict[str, Any]):
        """Display prediction-focused results"""
        symbol = result.get('symbol', 'Unknown')
        predictions = result.get('predictions', {})
        
        logger.debug(f"Displaying prediction results for {symbol}")
        logger.debug(f"Result keys: {list(result.keys())}")
        logger.debug(f"Predictions keys: {list(predictions.keys()) if predictions else 'None'}")
        
        # Title
        title_html = f"<div class='prediction-header'><h2>🔮 Price Prediction: {symbol}</h2></div>"
        self.results_column.append(pn.pane.HTML(title_html))
        
        # Prediction chart
        if 'chart_data' in result and predictions:
            logger.debug(f"Creating prediction chart for {symbol}")
            try:
                chart = self.visualizer.create_prediction_chart(
                    result['chart_data'],
                    predictions,
                    symbol
                )
                chart_pane = pn.pane.Plotly(chart, sizing_mode="stretch_width", height=500)
                self.results_column.append(chart_pane)
                logger.debug(f"Prediction chart created and added for {symbol}")
            except Exception as e:
                logger.error(f"Prediction chart creation failed: {e}", exc_info=True)
                # Add error display to the UI
                error_html = f"<div class='error'>Failed to create chart: {str(e)}</div>"
                self.results_column.append(pn.pane.HTML(error_html))
        else:
            logger.warning(f"Missing chart data or predictions for {symbol}. Chart data: {bool('chart_data' in result)}, Predictions: {bool(predictions)}")
        
        # Prediction summary
        pred_card = self._create_predictions_card(predictions)
        self.results_column.append(pred_card)
    
    def _display_comparison_results(self, result: Dict[str, Any]):
        """Display stock comparison results"""
        symbols = result.get('symbols', [])
        comparison_data = result.get('comparison_data', {})
        
        # Title
        title_html = f"<div class='comparison-header'><h2>⚖️ Stock Comparison: {' vs '.join(symbols)}</h2></div>"
        self.results_column.append(pn.pane.HTML(title_html))
        
        # Comparison chart
        if comparison_data:
            try:
                chart = self.visualizer.create_comparison_chart(comparison_data)
                chart_pane = pn.pane.Plotly(chart, sizing_mode="stretch_width", height=500)
                self.results_column.append(chart_pane)
            except Exception as e:
                logger.error(f"Comparison chart creation failed: {e}")
        
        # Summary table could be added here
    
    def _display_price_info(self, result: Dict[str, Any]):
        """Display current price information"""
        symbol = result.get('symbol', 'Unknown')
        current_data = result.get('current_data', {})
        stock_info = result.get('stock_info', {})
        
        # Title
        title_html = f"<div class='price-header'><h2>💰 Current Price: {symbol}</h2></div>"
        self.results_column.append(pn.pane.HTML(title_html))
        
        # Price card
        price_card = self._create_price_card(current_data, stock_info)
        self.results_column.append(price_card)
    
    def _display_error_result(self, result: Dict[str, Any]):
        """Display error message"""
        error_message = result.get('message', 'An unknown error occurred')
        
        error_html = f"""
        <div class='error-card'>
            <h3>❌ Error</h3>
            <p>{error_message}</p>
            <div class='suggestions'>
                <h4>Try these examples:</h4>
                <ul>
                    <li>Analyze AAPL</li>
                    <li>Predict GOOGL</li>
                    <li>Compare AAPL vs MSFT</li>
                    <li>TSLA current price</li>
                </ul>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(error_html))
    
    def _display_general_response(self, result: Dict[str, Any]):
        """Display general response"""
        message = result.get('message', 'Sorry, I could not process your request.')
        suggestions = result.get('suggestions', [])
        
        response_html = f"""
        <div class='general-response'>
            <h3>💡 How can I help?</h3>
            <p>{message}</p>
            <div class='suggestions'>
                <h4>Try these examples:</h4>
                <ul>
        """
        
        for suggestion in suggestions:
            response_html += f"<li>{suggestion}</li>"
        
        response_html += """
                </ul>
            </div>
        </div>
        """
        
        self.results_column.append(pn.pane.HTML(response_html))
    
    def _create_price_card(self, current_data: Dict[str, Any], stock_info: Dict[str, Any]) -> pn.pane.HTML:
        """Create price information card"""
        price = current_data.get('current_price', 0)
        prev_close = current_data.get('previous_close', 0)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        change_color = 'green' if change >= 0 else 'red'
        change_symbol = '▲' if change >= 0 else '▼'
        
        volume = current_data.get('volume', 0)
        market_cap = stock_info.get('market_cap')
        
        price_html = f"""
        <div class='price-card'>
            <div class='price-main'>
                <span class='current-price'>${price:.2f}</span>
                <span class='price-change' style='color: {change_color}'>
                    {change_symbol} ${abs(change):.2f} ({change_pct:+.2f}%)
                </span>
            </div>
            <div class='price-details'>
                <div class='detail-item'>
                    <span class='label'>Previous Close:</span>
                    <span class='value'>${prev_close:.2f}</span>
                </div>
                <div class='detail-item'>
                    <span class='label'>Volume:</span>
                    <span class='value'>{volume:,}</span>
                </div>
        """
        
        if market_cap:
            price_html += f"""
                <div class='detail-item'>
                    <span class='label'>Market Cap:</span>
                    <span class='value'>${market_cap/1e9:.1f}B</span>
                </div>
            """
        
        price_html += """
            </div>
        </div>
        """
        
        return pn.pane.HTML(price_html)
    
    def _create_signals_card(self, trading_signals: Dict[str, Any]) -> pn.pane.HTML:
        """Create trading signals card"""
        signal = trading_signals.get('primary_signal', 'HOLD')
        confidence = trading_signals.get('confidence', 0)
        recommendations = trading_signals.get('recommendations', [])
        
        signal_color = {
            'BUY': 'green',
            'SELL': 'red',
            'HOLD': 'orange'
        }.get(signal, 'gray')
        
        signals_html = f"""
        <div class='signals-card'>
            <h3>📈 Trading Signals</h3>
            <div class='signal-main'>
                <span class='signal-label'>Recommendation:</span>
                <span class='signal-value' style='color: {signal_color}'>{signal}</span>
                <span class='confidence'>({confidence:.1f}% confidence)</span>
            </div>
            <div class='recommendations'>
                <h4>Key Points:</h4>
                <ul>
        """
        
        for rec in recommendations[:3]:  # Show top 3 recommendations
            signals_html += f"<li>{rec}</li>"
        
        signals_html += """
                </ul>
            </div>
        </div>
        """
        
        return pn.pane.HTML(signals_html)
    
    def _create_predictions_card(self, predictions: Dict[str, Any]) -> pn.pane.HTML:
        """Create predictions summary card"""
        pred_values = predictions.get('predictions', [])
        last_price = predictions.get('last_price', 0)
        period_days = predictions.get('prediction_period_days', 30)
        
        if pred_values:
            avg_pred = sum(pred_values) / len(pred_values)
            max_pred = max(pred_values)
            min_pred = min(pred_values)
            
            price_change = avg_pred - last_price
            price_change_pct = (price_change / last_price * 100) if last_price else 0
            
            change_color = 'green' if price_change >= 0 else 'red'
            change_symbol = '▲' if price_change >= 0 else '▼'
        else:
            avg_pred = max_pred = min_pred = last_price
            price_change = price_change_pct = 0
            change_color = 'gray'
            change_symbol = '→'
        
        pred_html = f"""
        <div class='predictions-card'>
            <h3>🔮 AI Predictions ({period_days} days)</h3>
            <div class='prediction-summary'>
                <div class='pred-item'>
                    <span class='label'>Current Price:</span>
                    <span class='value'>${last_price:.2f}</span>
                </div>
                <div class='pred-item'>
                    <span class='label'>Predicted Average:</span>
                    <span class='value'>${avg_pred:.2f}</span>
                    <span class='change' style='color: {change_color}'>
                        {change_symbol} {price_change_pct:+.1f}%
                    </span>
                </div>
                <div class='pred-item'>
                    <span class='label'>Range:</span>
                    <span class='value'>${min_pred:.2f} - ${max_pred:.2f}</span>
                </div>
            </div>
            <div class='prediction-disclaimer'>
                <small>⚠️ Predictions are based on historical data and should not be used as sole investment advice.</small>
            </div>
        </div>
        """
        
        return pn.pane.HTML(pred_html)
    
    def _create_analysis_card(self, analysis_text: str) -> pn.pane.HTML:
        """Create AI analysis text card"""
        # Format the analysis text for better display
        formatted_text = analysis_text.replace('\n', '<br>')
        
        analysis_html = f"""
        <div class='analysis-card'>
            <h3>🤖 AI Analysis</h3>
            <div class='analysis-content'>
                {formatted_text}
            </div>
        </div>
        """
        
        return pn.pane.HTML(analysis_html)
    
    def _show_welcome(self):
        """Show welcome message"""
        welcome_html = """
        <div class='welcome-card'>
            <h2>🎯 Welcome to Stock Analysis AI</h2>
            <p>Get AI-powered stock analysis, predictions, and insights in natural language.</p>
            <div class='features'>
                <div class='feature'>
                    <h4>📊 Stock Analysis</h4>
                    <p>Get comprehensive technical analysis with trend signals</p>
                </div>
                <div class='feature'>
                    <h4>🔮 AI Predictions</h4>
                    <p>LSTM-based price forecasting with confidence intervals</p>
                </div>
                <div class='feature'>
                    <h4>⚖️ Stock Comparison</h4>
                    <p>Compare multiple stocks side by side</p>
                </div>
            </div>
            <div class='quick-start'>
                <h4>Quick Start:</h4>
                <p>Try: "Analyze AAPL" or "Predict GOOGL" or use the quick buttons above!</p>
            </div>
        </div>
        """
        self.results_column.append(pn.pane.HTML(welcome_html))
    
    def _show_error(self, message: str):
        """Show error message"""
        self.status_bar.object = f"<div class='status-error'>❌ {message}</div>"
        self._set_loading_state(False)
    
    def create_layout(self):
        """Create the main application layout"""
        
        # Custom CSS
        css = """
        <style>
        .query-input input { font-size: 16px; padding: 10px; }
        .status-bar { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status-ready { color: #666; }
        .status-loading { color: #007bff; }
        .status-success { color: #28a745; }
        .status-error { color: #dc3545; }
        .price-card { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0; }
        .price-main { display: flex; align-items: center; gap: 15px; margin-bottom: 15px; }
        .current-price { font-size: 24px; font-weight: bold; }
        .price-change { font-size: 16px; font-weight: bold; }
        .price-details { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .detail-item { display: flex; justify-content: space-between; }
        .label { color: #666; }
        .value { font-weight: bold; }
        .signals-card, .predictions-card, .analysis-card { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0; }
        .signal-main { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }
        .signal-value { font-size: 18px; font-weight: bold; }
        .confidence { color: #666; }
        .welcome-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin: 20px 0; }
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .feature { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; }
        .error-card { background: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border: 1px solid #f5c6cb; }
        .quick-buttons button { margin: 5px; }
        </style>
        """
        
        css_pane = pn.pane.HTML(css)
        
        # Header
        header = pn.pane.HTML("""
        <div style='background: linear-gradient(90deg, #1f2937 0%, #374151 100%); 
                    color: white; padding: 20px; text-align: center; margin-bottom: 20px; border-radius: 10px;'>
            <h1 style='margin: 0; font-size: 2.5em;'>📈 Stock Analysis AI</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.8;'>Powered by LSTM predictions and technical analysis</p>
        </div>
        """, sizing_mode="stretch_width")
        
        # Query section
        query_section = pn.Column(
            pn.Row(
                self.query_input,
                self.submit_button,
                self.loading_indicator,
                sizing_mode="stretch_width"
            ),
            self.quick_buttons,
            sizing_mode="stretch_width",
            margin=(10, 0)
        )
        
        # Main layout
        layout = pn.Column(
            css_pane,
            header,
            query_section,
            self.status_bar,
            self.results_column,
            sizing_mode="stretch_width",
            max_width=1200,
            margin=(0, 20)
        )
        
        return layout

def create_app():
    """Create and configure the Panel application"""
    app = StockAnalysisApp()
    layout = app.create_layout()
    
    template = pn.template.BootstrapTemplate(
        title="Stock Analysis AI",
        sidebar=[],
        header_background='#1f2937',
        sidebar_width=0
    )
    template.main.append(layout)
    
    return template.servable()