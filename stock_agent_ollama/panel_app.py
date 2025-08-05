"""
Stock Analysis AI - Panel Application
A comprehensive stock analysis system powered by Ollama (Gemma3), LangChain ReAct agents, 
LSTM ensemble neural networks, and Panel for enhanced user experience.

Converted from Streamlit with enhanced features:
- Real-time progress tracking
- Interactive Plotly charts  
- Expandable results sections
- Professional UI design
"""

import panel as pn
import threading
from datetime import datetime
import logging
import time
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Panel with extensions and custom CSS
pn.extension('plotly')

# Suppress noisy Panel/Bokeh warnings about patch dropping
import logging
bokeh_logger = logging.getLogger('bokeh')
bokeh_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

root_logger = logging.getLogger()
# Filter out the specific "Dropping a patch" warnings
class PatchWarningFilter(logging.Filter):
    def filter(self, record):
        return not ("Dropping a patch because it contains a previously known reference" in record.getMessage())

root_logger.addFilter(PatchWarningFilter())

# Add error suppression via template
pn.config.template = 'material'
plotly_error_script = """
<script>
// Suppress Plotly resize errors - console override
(function() {
    const originalError = console.error;
    console.error = function() {
        const message = Array.from(arguments).join(' ');
        if (!message.includes('Resize must be passed a displayed plot div element')) {
            originalError.apply(console, arguments);
        }
    };
    
    // Suppress unhandled promise rejections
    window.addEventListener('unhandledrejection', function(e) {
        if (e.reason && e.reason.message && 
            e.reason.message.includes('Resize must be passed a displayed plot div element')) {
            e.preventDefault();
        }
    });
})();
</script>
"""

# Add to Panel's template
pn.config.raw_css.append(plotly_error_script)

# Custom CSS for modern, native-like styling
CUSTOM_CSS = """
:root {
    --primary-color: #007AFF;
    --primary-hover: #0051D5;
    --success-color: #34C759;
    --warning-color: #FF9500;
    --error-color: #FF3B30;
    --background-color: #F2F2F7;
    --surface-color: #FFFFFF;
    --text-primary: #000000;
    --text-secondary: #6D6D80;
    --border-color: #E5E5EA;
    --shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    --border-radius: 12px;
    --border-radius-small: 8px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--background-color);
    color: var(--text-primary);
}

/* Panel container styling */
.bk-root .bk {
    background-color: var(--background-color);
}

/* Card-like containers */
.panel-card {
    background: var(--surface-color);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    border: 1px solid var(--border-color);
    padding: 20px;
    margin: 10px 0;
}

/* Modern button styling */
.bk-btn {
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius-small) !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}

.bk-btn:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px) !important;
}

.bk-btn:active {
    transform: translateY(0) !important;
}

/* Input styling */
.bk-input {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--border-radius-small) !important;
    padding: 12px 16px !important;
    font-size: 16px !important;
    transition: border-color 0.2s ease !important;
    background: var(--surface-color) !important;
}

.bk-input:focus {
    border-color: var(--primary-color) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1) !important;
}

/* Select dropdown styling */
.bk-input[readonly] {
    background: var(--surface-color) !important;
    cursor: pointer !important;
}

/* Progress bar styling */
.bk-progress {
    background: var(--border-color) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.bk-progress .bk-progress-fill {
    background: linear-gradient(90deg, var(--primary-color), var(--success-color)) !important;
    transition: width 0.3s ease !important;
}

/* Chat message styling */
.markdown-content {
    line-height: 1.6;
    font-size: 15px;
}

/* Accordion styling */
.bk-accordion-header {
    background: var(--surface-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--border-radius-small) !important;
    padding: 16px 20px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

.bk-accordion-content {
    background: var(--surface-color) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-radius: 0 0 var(--border-radius-small) var(--border-radius-small) !important;
    padding: 20px !important;
}

/* Scrollbar styling for webkit browsers */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--background-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}
"""

# Apply custom CSS
pn.config.raw_css.append(CUSTOM_CSS)


# Import existing components
from config import setup_logging, get_logger, get_config
from utils import extract_stock_symbols
from data_store import get_data_store

# Try to import stock_agent at module level
try:
    from stock_agent import StockAnalysisAgent
    STOCK_AGENT_AVAILABLE = True
    print("✅ StockAnalysisAgent imported successfully")
except ImportError as e:
    print(f"❌ Failed to import StockAnalysisAgent: {e}")
    STOCK_AGENT_AVAILABLE = False

# Setup logging
setup_logging()
logger = get_logger(__name__)

class ResultsDisplay:
    """Interactive results display with charts and metrics"""
    
    def __init__(self):
        self.data_store = get_data_store()
        self.create_results_components()
    
    def create_results_components(self):
        """Create results display components"""
        self.stock_info = pn.pane.Markdown("")
        self.charts_section = pn.Column()
        self.metrics_section = pn.Row()
        self.create_layout()
    
    def create_layout(self):
        """Create results layout with expandable accordion"""
        self.layout = pn.Accordion(
            ("📊 Analysis Results & Charts", pn.Column(
                self.stock_info,
                self.metrics_section,
                self.charts_section,
                styles={
                    'background': '#FFFFFF',
                    'padding': '16px',
                    'border-radius': '12px'
                }
            )),
            active=[0],  # Expanded by default
            width=640,  # Increased from 560 to 640 to match wider panel
            visible=False,
            styles={
                'border-radius': '12px',
                'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                'border': '1px solid #E5E5EA'
            }
        )
    
    def show_results(self, query):
        """Show results for the given query"""
        try:
            symbols = extract_stock_symbols(query)
            if not symbols:
                return
            
            symbol = symbols[0].upper()
            stock_data = self.data_store.get_stock_data(symbol)
            if not stock_data:
                logger.warning(f"No stock data found for {symbol}")
                return
            
            self._display_stock_info(symbol, stock_data)
            self._display_metrics(symbol, stock_data)
            self._display_charts(symbol)
            self.layout.visible = True
            
            logger.info(f"Results displayed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
    
    def _display_stock_info(self, symbol, stock_data):
        """Display basic stock information"""
        try:
            company_name = stock_data.get('company_name', 'Unknown')
            current_price = stock_data.get('current_price', 'N/A')
            data_range = stock_data.get('data_range', 'N/A')
            total_records = stock_data.get('total_records', 'N/A')
            
            info_text = f"""
            ### 📈 {symbol} - {company_name}
            
            **Current Price**: ${current_price:.2f}  
            **Data Period**: {data_range}  
            **Records Analyzed**: {total_records} data points
            """
            
            self.stock_info.object = info_text
            
        except Exception as e:
            logger.error(f"Error displaying stock info: {e}")
            self.stock_info.object = f"### 📈 {symbol}\nStock information available"
    
    def _display_metrics(self, symbol, stock_data):
        """Display summary metrics in card format"""
        try:
            # Get predictions data
            predictions_key = f"{symbol}_predictions"
            predictions = self.data_store.get_stock_data(predictions_key)
            
            if not predictions:
                predictions_key_alt = f"{symbol.upper()}_PREDICTIONS"
                predictions = self.data_store.get_stock_data(predictions_key_alt)
            
            # Create metric cards
            metrics = []
            
            # Current price card with modern styling
            current_price = stock_data.get('current_price')
            if current_price:
                price_card = pn.pane.HTML(f"""
                <div style='
                    background: linear-gradient(135deg, #007AFF, #0051D5); 
                    color: white;
                    padding: 20px; 
                    border-radius: 16px; 
                    text-align: center; 
                    margin: 8px;
                    box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
                    border: none;
                '>
                    <div style='font-size: 14px; font-weight: 600; margin-bottom: 8px; opacity: 0.9;'>💰 Current Price</div>
                    <div style='font-size: 28px; font-weight: 700; line-height: 1;'>${current_price:.2f}</div>
                </div>
                """, width=160)
                metrics.append(price_card)
            
            # Percentage change card
            percentage_change = 0
            if predictions and hasattr(predictions, 'iloc') and len(predictions) > 0:
                try:
                    latest_pred = predictions.iloc[-1]
                    # Try different possible column names for percentage change
                    percentage_change = latest_pred.get('Percentage_Change', 
                                      latest_pred.get('percentage_change',
                                      latest_pred.get('Trend_Percentage', 0)))
                    
                    # If still 0, calculate from prices
                    if percentage_change == 0:
                        current_price = stock_data.get('current_price', 0)
                        pred_price = latest_pred.get('Predicted_Price', 
                                    latest_pred.get('predicted_price', 0))
                        if current_price > 0 and pred_price > 0:
                            percentage_change = ((pred_price - current_price) / current_price) * 100
                            
                except Exception as e:
                    logger.debug(f"Error calculating percentage change: {e}")
                    percentage_change = 0
            
            # If still no data, try getting from trend analysis in predictions
            if percentage_change == 0 and predictions:
                try:
                    trend_analysis = predictions.get('trend_analysis', {})
                    percentage_change = trend_analysis.get('percentage_change', 0)
                    logger.debug(f"Percentage change from trend analysis: {percentage_change}")
                except Exception as e:
                    logger.debug(f"Error getting trend analysis: {e}")
                    percentage_change = 0
            
            # Debug logging
            logger.info(f"Final percentage change: {percentage_change}")
            if predictions and hasattr(predictions, 'iloc') and len(predictions) > 0:
                logger.debug(f"Prediction columns: {predictions.columns.tolist()}")
                logger.debug(f"Latest prediction data: {predictions.iloc[-1].to_dict()}")
            
            # Modern percentage change card with gradient
            if percentage_change >= 0:
                change_gradient = 'linear-gradient(135deg, #34C759, #30A14E)'
                change_shadow = 'rgba(52, 199, 89, 0.3)'
            else:
                change_gradient = 'linear-gradient(135deg, #FF3B30, #D70015)'
                change_shadow = 'rgba(255, 59, 48, 0.3)'
            
            change_icon = '📈' if percentage_change >= 0 else '📉'
            change_sign = '+' if percentage_change > 0 else ''
            
            change_card = pn.pane.HTML(f"""
            <div style='
                background: {change_gradient}; 
                color: white;
                padding: 20px; 
                border-radius: 16px; 
                text-align: center; 
                margin: 8px;
                box-shadow: 0 4px 12px {change_shadow};
                border: none;
            '>
                <div style='font-size: 14px; font-weight: 600; margin-bottom: 8px; opacity: 0.9;'>{change_icon} Change</div>
                <div style='font-size: 24px; font-weight: 700; line-height: 1;'>{change_sign}{percentage_change:.2f}%</div>
            </div>
            """, width=160)
            metrics.append(change_card)
            
            # Prediction cards
            if predictions and hasattr(predictions, 'iloc') and len(predictions) > 0:
                try:
                    latest_pred = predictions.iloc[-1]
                    pred_price = latest_pred.get('Predicted_Price')
                    if pred_price:
                        pred_card = pn.pane.HTML(f"""
                        <div style='
                            background: linear-gradient(135deg, #AF52DE, #8E44AD); 
                            color: white;
                            padding: 20px; 
                            border-radius: 16px; 
                            text-align: center; 
                            margin: 8px;
                            box-shadow: 0 4px 12px rgba(175, 82, 222, 0.3);
                            border: none;
                        '>
                            <div style='font-size: 14px; font-weight: 600; margin-bottom: 8px; opacity: 0.9;'>🔮 Prediction</div>
                            <div style='font-size: 24px; font-weight: 700; line-height: 1;'>${pred_price:.2f}</div>
                        </div>
                        """, width=160)
                        metrics.append(pred_card)
                    
                    # Modern trend card
                    trend = latest_pred.get('Trend', 'N/A')
                    trend_card = pn.pane.HTML(f"""
                    <div style='
                        background: linear-gradient(135deg, #FF9500, #FF7A00); 
                        color: white;
                        padding: 20px; 
                        border-radius: 16px; 
                        text-align: center; 
                        margin: 8px;
                        box-shadow: 0 4px 12px rgba(255, 149, 0, 0.3);
                        border: none;
                    '>
                        <div style='font-size: 14px; font-weight: 600; margin-bottom: 8px; opacity: 0.9;'>📊 Trend</div>
                        <div style='font-size: 18px; font-weight: 700; line-height: 1;'>{trend}</div>
                    </div>
                    """, width=160)
                    metrics.append(trend_card)
                    
                except Exception as pred_error:
                    logger.warning(f"Error processing predictions: {pred_error}")
            
            # Update metrics section
            self.metrics_section.clear()
            for metric in metrics:
                self.metrics_section.append(metric)
            
        except Exception as e:
            logger.error(f"Error displaying metrics: {e}")
    
    def _display_charts(self, symbol):
        """Display interactive Plotly charts"""
        try:
            viz_key = f"{symbol}_visualizations"
            viz_data = self.data_store.get_stock_data(viz_key)
            
            if not viz_data or not isinstance(viz_data, dict):
                logger.warning(f"No visualization data found for {symbol}")
                return
            
            charts = viz_data.get('chart_objects', {})
            self.charts_section.clear()
            
            # Display main price chart
            if 'main' in charts:
                self.charts_section.append(pn.pane.Markdown("### 📈 Price Chart with Predictions"))
                chart_pane = pn.pane.Plotly(
                    charts['main'], 
                    width=620, 
                    height=400,
                    config={'displayModeBar': True, 'responsive': True}
                )
                self.charts_section.append(chart_pane)
                logger.info(f"Added main chart for {symbol}")
            
            # Display volume chart
            if 'volume' in charts and charts['volume'] is not None:
                self.charts_section.append(pn.pane.Markdown("### 📊 Volume Analysis"))
                volume_pane = pn.pane.Plotly(
                    charts['volume'], 
                    width=620, 
                    height=300,
                    config={'displayModeBar': True, 'responsive': True}
                )
                self.charts_section.append(volume_pane)
                logger.info(f"Added volume chart for {symbol}")
            else:
                # Show message if volume data is not available
                self.charts_section.append(pn.pane.Markdown("### 📊 Volume Analysis"))
                self.charts_section.append(pn.pane.Markdown("Volume data not available for this stock."))
            
            # Display trend chart
            if 'trend' in charts:
                self.charts_section.append(pn.pane.Markdown("### 📉 Trend Analysis"))
                trend_pane = pn.pane.Plotly(
                    charts['trend'], 
                    width=620, 
                    height=300,
                    config={'displayModeBar': True, 'responsive': True}
                )
                self.charts_section.append(trend_pane)
                logger.info(f"Added trend chart for {symbol}")
            
            if not charts:
                self.charts_section.append(
                    pn.pane.Markdown("📊 **Charts will appear here after analysis completes**")
                )
            
        except Exception as e:
            logger.error(f"Error displaying charts: {e}")
            self.charts_section.clear()
            self.charts_section.append(
                pn.pane.Markdown(f"⚠️ **Error loading charts**: {str(e)}")
            )
    
    def hide_results(self):
        """Hide results display"""
        self.layout.visible = False
    
    def get_layout(self):
        """Return results layout"""
        return self.layout

class ProgressTracker:
    """Real-time progress tracking for analysis and LSTM training"""
    
    def __init__(self):
        self.create_progress_components()
        self.training_started = False
        self.create_layout()
    
    def create_progress_components(self):
        """Create progress tracking components"""
        self.progress_header = pn.pane.Markdown("### 🔄 Analysis Progress")
        
        # Main progress bar
        self.main_progress = pn.indicators.Progress(
            name="Analysis Steps",
            value=0,
            width=500,
            visible=False,
            bar_color="info",
            margin=(0, 0)  # Remove default margins
        )
        
        # Step indicator and details
        self.step_indicator = pn.pane.Markdown("Ready to start analysis...", margin=(0, 0))
        self.step_details = pn.pane.Markdown("", margin=(0, 0))
        
        # LSTM training progress section
        self.training_header = pn.pane.Markdown("**🧠 LSTM Ensemble Training:**")
        self.training_progress = pn.indicators.Progress(
            name="Neural Network Training",
            value=0,
            width=500,
            bar_color="success"
        )
        self.training_status = pn.pane.Markdown("")
        
        # Loss chart for training progress
        self.loss_chart = None
        self.loss_data = {"epochs": [], "train_loss": [], "val_loss": [], "model_num": []}
        self.current_model = None
        
        # Training container (hidden initially)
        self.training_container = pn.Column(
            self.training_header,
            self.training_progress,
            self.training_status,
            visible=False
        )
    
    def reset_progress(self):
        """Reset all progress tracking for a new analysis session"""
        logger.info("Resetting progress tracker - clearing completion messages")
        # Reset main progress
        self.main_progress.value = 0
        self.main_progress.visible = False
        self.step_indicator.object = "Ready to start analysis..."
        self.step_details.object = ""
        self.step_details.visible = False  # Hide the details component entirely during reset
        
        # Force UI update to ensure reset is applied
        try:
            self.step_details.param.trigger('object')
            self.step_details.param.trigger('visible')
            self.step_indicator.param.trigger('object')
        except:
            pass  # Ignore any trigger errors
        
        # Reset training progress
        self.training_progress.value = 0
        self.training_status.object = ""
        self.training_started = False
        self.training_container.visible = False
        
        # Reset loss data and chart
        self.loss_data = {"epochs": [], "train_loss": [], "val_loss": [], "model_num": []}
        self.current_model = None
        self.loss_chart = None
        
        # Clear any existing loss chart from training container
        if len(self.training_container.objects) > 3:  # Remove chart if it exists
            self.training_container.objects = self.training_container.objects[:3]
    
    def create_layout(self):
        """Create progress layout"""
        # Group progress bar and status together with no spacing
        progress_group = pn.Column(
            self.main_progress,
            self.step_indicator,
            self.step_details,
            margin=(0, 0)
        )
        
        self.layout = pn.Column(
            self.progress_header,
            progress_group,
            pn.Spacer(height=8),  # Small separation before training details
            self.training_container,
            visible=False,
            margin=(0, 0)
        )
    
    def show_progress(self):
        """Show progress tracking"""
        self.layout.visible = True
        self.main_progress.visible = True
    
    def hide_progress(self):
        """Hide progress tracking"""
        self.layout.visible = False
        self.main_progress.visible = False
        self.training_container.visible = False
        self.training_started = False
    
    def update_main_progress(self, step, total, message):
        """Update main analysis progress"""
        try:
            progress_value = int((step / total) * 100) if total > 0 else 0
            self.main_progress.value = progress_value
            self.main_progress.visible = True  # Ensure progress bar is visible when updating
            
            # Use the actual message from the agent instead of hardcoded descriptions
            if message:
                step_name = message  # Use the actual agent message
            else:
                step_name = f"🔄 Processing Step {step}"
            
            self.step_indicator.object = f"**{step_name}** ({step}/{total})"
            self.step_details.object = ""  # Clear details since message is self-descriptive
            
            # Show training progress when LSTM starts (detect by message content)
            if message and ("Training LSTM" in message or "🧠" in message) and not self.training_started:
                self.training_container.visible = True
                self.training_started = True
                self.training_progress.value = 0
                self.training_status.object = "Initializing neural networks..."
                # Reset loss data for new training session
                self.loss_data = {"epochs": [], "train_loss": [], "val_loss": [], "model_num": []}
                self.loss_chart = None
                self.current_model = None
            
            logger.info(f"Progress update: {step}/{total} - {message}")
            
        except Exception as e:
            logger.error(f"Progress update failed: {e}")
    
    def update_training_progress(self, message):
        """Update LSTM training progress with detailed info"""
        try:
            if not self.training_started:
                return
            
            model_info = ""
            epoch_info = ""
            
            # Extract model information (format: "Training Model 1/3")
            if "Training Model" in message:
                if "1/3" in message:
                    model_info = "🔵 Training Model 1 of 3"
                elif "2/3" in message:
                    model_info = "🟡 Training Model 2 of 3"
                elif "3/3" in message:
                    model_info = "🟢 Training Model 3 of 3"
            
            # Extract epoch and loss information (format: "Epoch 10/75")
            if "Epoch" in message and "/" in message:
                try:
                    # Parse format: "🧠 Training Model 1/3 • Epoch 10/75 • Loss: 0.0123 • Val Loss: 0.0456"
                    epoch_part = message.split("Epoch")[1].split("•")[0].strip()
                    if "/" in epoch_part:
                        current, total_epochs = map(int, epoch_part.split("/"))
                        epoch_progress = current / total_epochs
                        
                        # Calculate overall training progress across all 3 models
                        if "1/3" in message:
                            overall_progress = epoch_progress / 3
                        elif "2/3" in message:
                            overall_progress = (1 + epoch_progress) / 3
                        elif "3/3" in message:
                            overall_progress = (2 + epoch_progress) / 3
                            # Check if this is the final epoch of the final model
                            if current == total_epochs:
                                self.complete_training_progress()
                                return  # Don't update progress again, training is complete
                        else:
                            overall_progress = epoch_progress
                        
                        progress_percent = int(overall_progress * 100)
                        self.training_progress.value = progress_percent
                        epoch_info = f"Epoch {current}/{total_epochs} - Overall: {progress_percent}%"
                        
                        # Extract loss information if available (format: "Loss: 0.0123 • Val Loss: 0.0456")
                        if "Loss:" in message and "Val Loss:" in message:
                            try:
                                loss_part = message.split("Loss:")[1].split("•")[0].strip()
                                val_loss_part = message.split("Val Loss:")[1].strip()
                                epoch_info += f"\nTrain Loss: {loss_part} | Val Loss: {val_loss_part}"
                                
                                # Parse numerical values for chart
                                train_loss_val = float(loss_part)
                                val_loss_val = float(val_loss_part)
                                
                                # Determine model number
                                if "1/3" in message:
                                    model_num = 1
                                elif "2/3" in message:
                                    model_num = 2
                                elif "3/3" in message:
                                    model_num = 3
                                else:
                                    model_num = 1
                                
                                # Update loss chart
                                self.update_loss_chart(current, train_loss_val, val_loss_val, model_num)
                                
                            except Exception as loss_e:
                                logger.debug(f"Could not parse loss info: {loss_e}")
                    
                except Exception as e:
                    logger.warning(f"Could not parse epoch info: {e}")
            
            # Update training status
            status_text = ""
            if model_info:
                status_text += model_info
            if epoch_info:
                if status_text:
                    status_text += "\n"
                status_text += epoch_info
            
            if status_text:
                self.training_status.object = status_text
                logger.debug(f"Training status updated: {status_text}")
            
        except Exception as e:
            logger.error(f"Training progress update failed: {e}")
    
    def complete_progress(self, message="Analysis completed successfully!"):
        """Mark progress as complete"""
        logger.info(f"Setting completion message: {message}")
        self.main_progress.value = 100
        self.step_indicator.object = f"✅ **{message}**"
        self.step_details.object = "All steps completed successfully!"
        self.step_details.visible = True  # Make details visible for completion message
        
        # Don't automatically set training to 100% - let training completion handle that
        # Training progress should only be completed when we actually finish all 3 models
    
    def complete_training_progress(self):
        """Mark training progress as complete"""
        if self.training_started:
            self.training_progress.value = 100
            self.training_status.object = "🎉 All 3 models trained successfully!"
    
    def create_loss_chart(self):
        """Create initial loss chart"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add empty traces for train and validation loss
        fig.add_trace(go.Scatter(
            x=[], y=[], 
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=[], y=[], 
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Training Loss Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template='plotly_white',
            height=250,
            width=550,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def update_loss_chart(self, epoch, train_loss, val_loss, model_num):
        """Update loss chart with new data point"""
        try:
            # Check if we're starting a new model - if so, clear the data
            if self.current_model is not None and self.current_model != model_num:
                logger.info(f"Clearing loss chart for new model: {self.current_model} -> {model_num}")
                self.loss_data = {"epochs": [], "train_loss": [], "val_loss": [], "model_num": []}
            
            # Update current model
            self.current_model = model_num
            
            # Add data point
            self.loss_data["epochs"].append(epoch)
            self.loss_data["train_loss"].append(train_loss)
            self.loss_data["val_loss"].append(val_loss)
            self.loss_data["model_num"].append(model_num)
            
            # Create/update chart
            if self.loss_chart is None:
                fig = self.create_loss_chart()
                self.loss_chart = pn.pane.Plotly(fig, width=550, height=250)
                # Add chart to training container
                if len(self.training_container.objects) == 3:  # header, progress, status
                    self.training_container.append(pn.Spacer(height=10))
                    self.training_container.append(self.loss_chart)
            
            # Update chart title to show current model
            fig = self.loss_chart.object
            fig.update_layout(title=f"Training Loss Progress - Model {model_num}/3")
            
            # Update chart data
            fig.data[0].x = self.loss_data["epochs"]
            fig.data[0].y = self.loss_data["train_loss"]
            fig.data[1].x = self.loss_data["epochs"] 
            fig.data[1].y = self.loss_data["val_loss"]
            
            # Update the chart
            self.loss_chart.param.trigger('object')
            
        except Exception as e:
            logger.error(f"Error updating loss chart: {e}")
    
    def get_layout(self):
        """Return progress layout"""
        return self.layout

class StockSidebar:
    """Left sidebar with quick analysis controls, templates, and chat input"""
    
    def __init__(self):
        self.create_sidebar_components()
    
    def create_sidebar_components(self):
        """Create sidebar components"""
        # Chat Input section
        self.chat_header = pn.pane.Markdown("## 💬 Ask AI Assistant")
        
        self.chat_input = pn.widgets.TextAreaInput(
            name="Your question:",
            placeholder="e.g., 'Analyze AAPL stock...'",
            height=80,
            sizing_mode='stretch_width'  # Allow to expand within column
        )
        
        # Add Enter key support for better UX
        self.chat_input.param.watch(self._on_chat_enter, 'value')
        
        # Add a watcher to track value changes
        self.chat_input.param.watch(self._on_chat_input_change, 'value')
        self._last_chat_value = ""
        
        self.send_btn = pn.widgets.Button(
            name="Send",
            button_type="primary",
            sizing_mode='stretch_width'  # Allow to expand within column
        )
        
        # Divider
        self.divider1 = pn.pane.HTML("<hr>")
        
        # Quick Analysis section
        self.quick_analysis_header = pn.pane.Markdown("## 🔍 Quick Analysis")
        
        self.quick_symbol = pn.widgets.TextInput(
            name="Stock Symbol",
            placeholder="e.g., AAPL",
            sizing_mode='stretch_width'  # Allow to expand within column
        )
        
        # Analysis parameters with responsive sizing
        self.past_data_period = pn.widgets.Select(
            name="Past Data",
            options=["2y", "5y"],
            value="2y",
            sizing_mode='stretch_width'  # Make responsive
        )
        
        self.future_prediction = pn.widgets.Select(
            name="Prediction", 
            options=["30 days", "1 week"],
            value="30 days",
            sizing_mode='stretch_width'  # Make responsive
        )
        
        self.params_row = pn.Row(
            self.past_data_period,
            pn.Spacer(width=10),  # Add spacing between dropdowns
            self.future_prediction,
            sizing_mode='stretch_width'  # Make the row responsive too
        )
        
        self.quick_analysis_btn = pn.widgets.Button(
            name="🔍 Quick Analysis",
            button_type="primary",
            sizing_mode='stretch_width'  # Allow to expand within column
        )
        
        # Divider
        self.divider2 = pn.pane.HTML("<hr>")
        
        # Analysis Templates section
        self.templates_header = pn.pane.Markdown("## 📊 Templates")
        
        self.template_options = {
            "Basic Analysis": "basic_analysis",
            "Trend Analysis": "trend_analysis", 
            "Risk Assessment": "risk_assessment"
        }
        
        self.selected_template = pn.widgets.Select(
            name="Choose template:",
            options=list(self.template_options.keys()),
            value="Basic Analysis",
            sizing_mode='stretch_width'  # Allow to expand within column
        )
        
        self.template_symbol = pn.widgets.TextInput(
            name="Symbol for template",
            placeholder="e.g., AAPL",
            sizing_mode='stretch_width'  # Allow to expand within column
        )
        
        self.template_btn = pn.widgets.Button(
            name="📈 Use Template",
            button_type="primary",
            sizing_mode='stretch_width'  # Allow to expand within column
        )
        
        # Bind events
        self.bind_events()
        self.create_layout()
    
    def bind_events(self):
        """Bind button events"""
        self.send_btn.on_click(self.on_send_message)
        self.quick_analysis_btn.on_click(self.on_quick_analysis)
        self.template_btn.on_click(self.on_template_analysis)
    
    def _on_chat_input_change(self, event):
        """Track chat input changes to handle timing issues"""
        self._last_chat_value = event.new if event.new else ""
        logger.info(f"Chat input watcher - value changed to: '{self._last_chat_value}'")
    
    def _on_chat_enter(self, event):
        """Handle Enter key in chat input"""
        if event.new and event.new.endswith('\n'):
            # User pressed Enter - strip newline and process
            message = event.new.rstrip('\n').strip()
            if message:
                logger.info(f"Enter key detected, processing: '{message}'")
                if hasattr(self, 'chat_interface'):
                    self.chat_interface.process_user_query(message)
                    self.chat_input.value = ""  # Clear input
    
    def on_send_message(self, event):
        """Handle send button click"""
        # Enhanced debugging for timing issues
        raw_value = self.chat_input.value
        logger.info(f"Send clicked - raw chat_input.value: {repr(raw_value)}")
        
        message = raw_value.strip() if raw_value else ""
        logger.info(f"Processed message: '{message}'")
        
        # Retry mechanism for timing issues
        if not message:
            # Try to get value again with a longer delay to account for Panel's reactivity
            try:
                import time
                time.sleep(0.2)  # Longer delay - 200ms to allow Panel to sync
                retry_value = self.chat_input.value
                logger.info(f"Retry value after delay: {repr(retry_value)}")
                message = retry_value.strip() if retry_value else ""
                
                # Also check the watcher value after the delay
                if not message and hasattr(self, '_last_chat_value') and self._last_chat_value:
                    logger.info(f"Using fallback value from watcher after delay: '{self._last_chat_value}'")
                    message = self._last_chat_value.strip()
            except:
                pass
        
        if message:
            if hasattr(self, 'chat_interface'):
                self.chat_interface.process_user_query(message)
                self.chat_input.value = ""  # Clear input
        else:
            logger.warning(f"No message entered. Final value: '{self.chat_input.value}'")
            if hasattr(self, 'chat_interface'):
                self.chat_interface.add_system_message("⚠️ Please enter a question first")
    
    def on_quick_analysis(self, event):
        """Handle quick analysis button click"""
        # Enhanced debugging
        logger.info(f"Widget object: {self.quick_symbol}")
        logger.info(f"Widget attributes: {dir(self.quick_symbol)}")
        
        raw_value = self.quick_symbol.value
        value_input = getattr(self.quick_symbol, 'value_input', '')
        
        logger.info(f"Raw quick_symbol.value: {repr(raw_value)}")
        logger.info(f"quick_symbol.value_input: {repr(value_input)}")
        
        # Use value_input as fallback if value is empty (race condition fix)
        symbol = raw_value.strip() if raw_value else value_input.strip() if value_input else ""
        logger.info(f"Processed symbol: '{symbol}'")
        
        if symbol:
            past_data = self.past_data_period.value
            future_pred = self.future_prediction.value
            prediction_days = 30 if future_pred == "30 days" else 7
            
            query = f"Analyze {symbol.upper()} stock using {past_data} of historical data and predict {prediction_days} days ahead"
            
            logger.info(f"Generated query: {query}")
            
            if hasattr(self, 'chat_interface'):
                self.chat_interface.process_user_query(query)
        else:
            logger.warning(f"No symbol entered. Raw value: '{self.quick_symbol.value}'")
            if hasattr(self, 'chat_interface'):
                self.chat_interface.add_system_message(f"⚠️ Please enter a stock symbol first (current value: '{self.quick_symbol.value}')")
    
    def on_template_analysis(self, event):
        """Handle template analysis button click"""
        # Fix race condition similar to quick analysis
        raw_value = self.template_symbol.value
        value_input = getattr(self.template_symbol, 'value_input', '')
        
        logger.info(f"Template symbol - raw_value: {repr(raw_value)}, value_input: {repr(value_input)}")
        
        # Use value_input as fallback if value is empty (race condition fix)
        symbol = raw_value.strip() if raw_value else value_input.strip() if value_input else ""
        
        if symbol:
            template_name = self.selected_template.value
            template_key = self.template_options[template_name]
            
            templates = {
                "basic_analysis": f"Analyze {symbol.upper()} stock and provide insights on its current performance and future outlook",
                "trend_analysis": f"Focus on the trending patterns of {symbol.upper()} stock and predict short-term movements",
                "risk_assessment": f"Evaluate the risk profile of investing in {symbol.upper()} stock based on historical data"
            }
            
            query = templates.get(template_key, f"Analyze {symbol.upper()} stock")
            
            if hasattr(self, 'chat_interface'):
                self.chat_interface.process_user_query(query)
        else:
            if hasattr(self, 'chat_interface'):
                self.chat_interface.add_system_message("⚠️ Please enter a symbol for template")
    
    def set_chat_interface(self, chat_interface):
        """Set reference to chat interface"""
        self.chat_interface = chat_interface
    
    def create_layout(self):
        """Create layout components"""
        # Combined bottom controls with modern card styling
        self.bottom_controls = pn.Row(
            # Ask AI Assistant section
            pn.Column(
                self.chat_header,
                self.chat_input,
                self.send_btn,
                margin=(10, 10),
                sizing_mode='stretch_width',
                styles={
                    'background': '#FFFFFF',
                    'border-radius': '12px',
                    'border': '1px solid #E5E5EA',
                    'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                    'padding': '20px'
                }
            ),
            pn.Spacer(width=15),  # Spacing between cards
            # Quick Analysis section
            pn.Column(
                self.quick_analysis_header,
                self.quick_symbol,
                self.params_row,
                self.quick_analysis_btn,
                margin=(10, 10),
                sizing_mode='stretch_width',
                styles={
                    'background': '#FFFFFF',
                    'border-radius': '12px',
                    'border': '1px solid #E5E5EA',
                    'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                    'padding': '20px'
                }
            ),
            pn.Spacer(width=15),  # Spacing between cards
            # Templates section  
            pn.Column(
                self.templates_header,
                self.selected_template,
                self.template_symbol,
                self.template_btn,
                margin=(10, 10),
                sizing_mode='stretch_width',
                styles={
                    'background': '#FFFFFF',
                    'border-radius': '12px',
                    'border': '1px solid #E5E5EA',
                    'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                    'padding': '20px'
                }
            ),
            margin=(15, 15),
            sizing_mode='stretch_width'
        )
        
        # Keep separate references for backward compatibility
        self.top_controls = pn.Spacer()  # Empty spacer since controls moved to bottom
        self.chat_input_section = pn.Spacer()  # Empty spacer since moved to bottom_controls
        
        # For backward compatibility, keep the full layout
        self.layout = pn.Column(
            self.top_controls,
            self.chat_input_section,
            width=540
        )
    
    def get_layout(self):
        return self.layout
    
    def get_top_controls(self):
        """Get the top controls (Quick Analysis + Templates)"""
        return self.top_controls
    
    def get_chat_input_section(self):
        """Get the chat input section"""
        return self.chat_input_section
    
    def get_bottom_controls(self):
        """Get all bottom controls combined (Ask AI + Quick Analysis + Templates)"""
        return self.bottom_controls

class ChatInterface:
    """Main chat interface with agent integration"""
    
    def __init__(self):
        self.messages = []
        self.agent = None  # Lazy load agent
        self.analysis_running = False
        self.create_chat_components()
    
    def create_chat_components(self):
        """Create chat interface components"""
        # Status indicator
        self.status = pn.pane.Markdown("")
        
        # Chat history container with enhanced styling
        self.chat_history = pn.Column(
            height=450,
            scroll=True,
            auto_scroll_limit=1,
            scroll_position=1,
            width_policy='max',
            sizing_mode='stretch_width',
            styles={
                'background': '#FFFFFF',
                'border-radius': '12px',
                'border': '1px solid #E5E5EA',
                'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                'padding': '16px',
                'margin': '8px 0'
            }
        )
        
        # Add initial welcome message
        self.add_assistant_message(
            "Hello! I'm your AI Stock Analysis Assistant powered by advanced machine learning.\n\n"
            "**What I can do:**\n\n"
            "📊 Analyze stocks with real-time data\n"
            "🧠 Generate LSTM price predictions\n"
            "📈 Create interactive charts\n"
            "💡 Provide comprehensive analysis\n"
            "📉 Assess investment risks\n\n"
            "**Try asking:**\n\n"
            "'Analyze AAPL stock'\n"
            "'Predict TSLA trends for 30 days'\n"
            "'Compare GOOGL vs MSFT'\n\n"
            "Use the input controls below to get started!"
        )
        
        self.create_layout()
    
    def process_user_query(self, query):
        """Process user query with progress tracking"""
        if self.analysis_running:
            self.add_system_message("⚠️ Analysis already in progress. Please wait...")
            return
        
        # Add user message
        self.add_user_message(query)
        
        # Hide previous results and show progress
        if hasattr(self, 'external_results_display') and self.external_results_display:
            self.external_results_display.hide_results()
        
        if hasattr(self, 'external_progress_tracker') and self.external_progress_tracker:
            # Reset BEFORE showing to ensure clean state
            self.external_progress_tracker.reset_progress()
            self.external_progress_tracker.show_progress()
        
        # Start analysis
        self.start_analysis(query)
    
    def start_analysis(self, query):
        """Start analysis with progress tracking"""
        self.analysis_running = True
        
        # Add initializing message to chat
        self.add_assistant_message("🔄 **Initializing analysis...** Please wait while I process your request.")
        
        # Start in background thread
        thread = threading.Thread(target=self._run_analysis, args=(query,), daemon=True)
        thread.start()
    
    def _run_analysis(self, query):
        """Run analysis in background thread with progress updates"""
        try:
            # Lazy load agent
            if self.agent is None:
                logger.info("Lazy loading StockAnalysisAgent...")
                
                if STOCK_AGENT_AVAILABLE:
                    self.agent = StockAnalysisAgent()
                    # Set progress callback for real-time updates
                    self.agent.set_progress_callback(self._agent_progress_callback)
                    logger.info("Agent loaded successfully")
                else:
                    raise Exception("StockAnalysisAgent is not available - check imports")
            
            logger.info(f"Running analysis for: {query}")
            
            # Run agent analysis
            result = self.agent.analyze_stock(query)
            
            # Handle result
            self._handle_analysis_result(result, query)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self._handle_analysis_error(str(e))
    
    def _agent_progress_callback(self, step, total, message):
        """Progress callback from agent"""
        try:
            self._update_progress_ui(step, total, message)
        except Exception as e:
            logger.error(f"Progress callback failed: {e}")
    
    def _update_progress_ui(self, step, total, message):
        """Update progress UI in chat window"""
        try:
            # Check if this is detailed training progress (epoch-level details)
            is_detailed_training = any(keyword in message for keyword in ["Training Model", "Epoch", "Loss:"])
            
            # Show main progress steps in chat (including LSTM step), but not detailed epoch training
            if not is_detailed_training:
                if step and total:
                    progress_percent = int((step / total) * 100)
                    progress_msg = f"**Step {step}/{total}** ({progress_percent}%) - {message}"
                else:
                    progress_msg = message
                
                # Add progress update to chat
                self.add_assistant_message(f"📊 {progress_msg}")
            
            # Always update external progress tracker (for right sidebar)
            if hasattr(self, 'external_progress_tracker') and self.external_progress_tracker:
                # Only send high-level step messages to main progress, detailed training to training progress
                if is_detailed_training:
                    # Send detailed training info only to training progress tracker
                    self.external_progress_tracker.update_training_progress(message)
                    logger.debug(f"Routing detailed training message: {message}")
                else:
                    # Send high-level step messages to main progress tracker
                    self.external_progress_tracker.update_main_progress(step, total, message)
                
        except Exception as e:
            logger.error(f"Progress UI update failed: {e}")
    
    def _handle_analysis_result(self, result, query):
        """Handle analysis result"""
        try:
            self.analysis_running = False
            
            if result.get("success", False):
                response = result.get("response", "Analysis completed successfully.")
                
                # Always show the main response in chat
                self.add_assistant_message(response)
                
                # Debug: Log what we actually received
                logger.info(f"Agent result keys: {list(result.keys())}")
                logger.info(f"Agent response: {response[:200]}...")
                
                # Try to extract insights from the response text itself
                if "Generated 3 key insights" in response or "key insights" in response.lower():
                    # Try to get insights from data store
                    from utils import extract_stock_symbols
                    symbols = extract_stock_symbols(query)
                    if symbols:
                        symbol = symbols[0].upper()
                        
                        # Try to get insights from data store
                        data_store = get_data_store()
                        
                        # Look for prediction data
                        predictions_key = f"{symbol}_predictions"
                        predictions = data_store.get_stock_data(predictions_key)
                        
                        # Look for stock data
                        stock_data = data_store.get_stock_data(symbol)
                        
                        # Create insights from available data
                        insights_parts = []
                        
                        if stock_data:
                            current_price = stock_data.get('current_price')
                            if current_price:
                                insights_parts.append(f"💰 **Current Price**: ${current_price:.2f}")
                            
                            company_name = stock_data.get('company_name', symbol)
                            data_range = stock_data.get('data_range', 'N/A')
                            insights_parts.append(f"📊 **Analysis Period**: {data_range} of {company_name} data")
                        
                        if predictions and hasattr(predictions, 'iloc') and len(predictions) > 0:
                            try:
                                latest_pred = predictions.iloc[-1]
                                pred_price = latest_pred.get('Predicted_Price')
                                trend = latest_pred.get('Trend', 'N/A')
                                
                                if pred_price and current_price:
                                    change_pct = ((pred_price - current_price) / current_price) * 100
                                    direction = "📈 Bullish" if change_pct > 0 else "📉 Bearish"
                                    insights_parts.append(f"🔮 **Prediction**: ${pred_price:.2f} ({change_pct:+.1f}%)")
                                    insights_parts.append(f"📈 **Trend Analysis**: {direction} momentum detected")
                                
                                if trend != 'N/A':
                                    insights_parts.append(f"🎯 **Technical Trend**: {trend}")
                                
                            except Exception as e:
                                logger.debug(f"Error extracting prediction insights: {e}")
                        
                        # If we have insights, display them
                        if insights_parts:
                            insights_text = "## 📋 **Key Insights:**\n\n" + "\n".join(insights_parts)
                            self.add_assistant_message(insights_text)
                        else:
                            logger.warning("No insights data found to display")
                
                # Legacy support: try to extract insights from result structure
                insights = result.get("insights", [])
                analysis_data = result.get("analysis_data", {})
                
                # If we have detailed insights, show them
                if insights and len(insights) > 0:
                    insights_text = "## 📋 **Key Insights:**\n\n"
                    for i, insight in enumerate(insights, 1):
                        insights_text += f"**{i}.** {insight}\n\n"
                    self.add_assistant_message(insights_text)
                
                # If we have analysis data with summary, show it
                elif analysis_data:
                    summary_parts = []
                    
                    # Extract key metrics if available
                    if 'current_price' in analysis_data:
                        summary_parts.append(f"💰 **Current Price**: ${analysis_data['current_price']:.2f}")
                    
                    if 'price_change' in analysis_data:
                        change = analysis_data['price_change']
                        direction = "📈" if change >= 0 else "📉"
                        summary_parts.append(f"{direction} **Price Change**: {change:+.2f}%")
                    
                    if 'volatility' in analysis_data:
                        vol = analysis_data['volatility']
                        summary_parts.append(f"📊 **Volatility**: {vol:.2f}%")
                    
                    if 'trend' in analysis_data:
                        summary_parts.append(f"📈 **Trend**: {analysis_data['trend']}")
                    
                    if summary_parts:
                        summary_text = "## 📊 **Analysis Summary:**\n\n" + "\n".join(summary_parts)
                        self.add_assistant_message(summary_text)
                
                # Handle progress completion based on query type
                if hasattr(self, 'external_progress_tracker') and self.external_progress_tracker:
                    # Check if this was a general question (no stock symbols in query)
                    from utils import extract_stock_symbols
                    symbols = extract_stock_symbols(query)
                    is_general_question = len(symbols) == 0
                    
                    if is_general_question:
                        # For general questions, just hide progress without showing completion
                        def hide_progress_delayed():
                            time.sleep(1)  # Shorter delay for general questions
                            self.external_progress_tracker.hide_progress()
                        
                        threading.Thread(target=hide_progress_delayed, daemon=True).start()
                    else:
                        # For stock analysis, show completion then hide
                        self.external_progress_tracker.complete_progress("Analysis completed!")
                        
                        def hide_progress_delayed():
                            time.sleep(3)
                            self.external_progress_tracker.hide_progress()
                        
                        threading.Thread(target=hide_progress_delayed, daemon=True).start()
                
                if hasattr(self, 'external_results_display') and self.external_results_display:
                    self.external_results_display.show_results(query)
                
            else:
                error_msg = result.get("error", "Unknown error occurred")
                self.add_assistant_message(f"❌ **Analysis Failed**: {error_msg}")
                
                if hasattr(self, 'external_progress_tracker') and self.external_progress_tracker:
                    self.external_progress_tracker.hide_progress()
            
            logger.info("Analysis result handled successfully")
            
        except Exception as e:
            logger.error(f"Error handling analysis result: {e}")
            self._handle_analysis_error(str(e))
    
    def _handle_analysis_error(self, error):
        """Handle analysis error"""
        self.analysis_running = False
        self.add_assistant_message(f"❌ **Error**: {error}")
        
        if hasattr(self, 'external_progress_tracker') and self.external_progress_tracker:
            self.external_progress_tracker.hide_progress()
            
        logger.error(f"Analysis error: {error}")
    
    def set_external_components(self, progress_tracker, results_display):
        """Set external progress tracker and results display components"""
        self.external_progress_tracker = progress_tracker
        self.external_results_display = results_display
    
    def add_user_message(self, message):
        """Add user message to chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        user_msg = pn.pane.Markdown(
            f"**👤 You ({timestamp})**\n\n{message}",
            styles={
                'background': 'linear-gradient(135deg, #007AFF, #0051D5)',
                'color': 'white',
                'padding': '16px 20px',
                'margin': '8px 0',
                'border-radius': '18px 18px 4px 18px',
                'width': '100%',
                'min-width': '100%',
                'box-sizing': 'border-box',
                'display': 'block',
                'box-shadow': '0 2px 8px rgba(0, 122, 255, 0.3)',
                'font-size': '15px',
                'line-height': '1.5'
            },
            width_policy='max',
            sizing_mode='stretch_width',
            margin=(5, 0)
        )
        
        self.chat_history.append(user_msg)
        self.messages.append({"role": "user", "content": message, "timestamp": timestamp})
    
    def add_assistant_message(self, message):
        """Add assistant message to chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        assistant_msg = pn.pane.Markdown(
            f"**🤖 Assistant ({timestamp})**\n\n{message}",
            styles={
                'background': '#F8F9FA',
                'border': '1px solid #E5E5EA',
                'padding': '16px 20px',
                'margin': '8px 0',
                'border-radius': '18px 18px 18px 4px',
                'width': '100%',
                'min-width': '100%',
                'box-sizing': 'border-box',
                'display': 'block',
                'overflow-wrap': 'break-word',
                'box-shadow': '0 1px 4px rgba(0, 0, 0, 0.1)',
                'font-size': '15px',
                'line-height': '1.5',
                'color': '#1D1D1F'
            },
            width_policy='max',
            sizing_mode='stretch_width',
            margin=(5, 0)
        )
        
        self.chat_history.append(assistant_msg)
        self.messages.append({"role": "assistant", "content": message, "timestamp": timestamp})
    
    def add_system_message(self, message):
        """Add system message to chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        system_msg = pn.pane.Markdown(
            f"**ℹ️ System ({timestamp})**\n\n{message}",
            styles={
                'background': 'linear-gradient(135deg, #FF9500, #FF7A00)',
                'color': 'white',
                'padding': '12px 16px',
                'margin': '8px 0',
                'border-radius': '10px',
                'width': '100%',
                'min-width': '100%',
                'box-sizing': 'border-box',
                'display': 'block',
                'box-shadow': '0 2px 6px rgba(255, 149, 0, 0.3)',
                'font-size': '14px',
                'line-height': '1.4',
                'text-align': 'center'
            },
            width_policy='max',
            sizing_mode='stretch_width',
            margin=(5, 0)
        )
        
        self.chat_history.append(system_msg)
        self.messages.append({"role": "system", "content": message, "timestamp": timestamp})
    
    def create_layout(self):
        """Create chat layout"""
        # Chat header (will be updated with toggle button in main app)
        self.header = pn.pane.Markdown("## 💬 Chat with AI Assistant")
        
        self.layout = pn.Column(
            self.header,
            self.status,
            self.chat_history,
            width_policy='max',
            sizing_mode='stretch_width'
        )
    
    def set_header_with_toggle(self, toggle_button):
        """Update header to include toggle button"""
        self.header_row = pn.Row(
            pn.pane.Markdown("## 💬 Chat with AI Assistant"),
            pn.Spacer(sizing_mode='stretch_width'),
            toggle_button,
            styles={
                'align-items': 'center',
                'margin-bottom': '10px'
            }
        )
        # Replace the header in the layout
        self.layout[0] = self.header_row
    
    def get_layout(self):
        return self.layout


class RightSidebar:
    """Right sidebar with progress tracking and results display"""
    
    def __init__(self):
        self.progress_tracker = ProgressTracker()
        self.results_display = ResultsDisplay()
        self.create_layout()
    
    def create_layout(self):
        """Create right sidebar layout"""
        self.layout = pn.Column(
            pn.pane.Markdown("## 📊 Analysis Status & Results"),
            pn.Spacer(height=10),
            self.progress_tracker.get_layout(),
            pn.Spacer(height=20),
            self.results_display.get_layout(),
            width=680,  # Increased from 580 to 680
            margin=(5, 5),
            styles={
                'background': '#FFFFFF',
                'border-radius': '12px',
                'border': '1px solid #E5E5EA',
                'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                'padding': '20px'
            }
        )
    
    def get_layout(self):
        return self.layout
    
    def get_progress_tracker(self):
        return self.progress_tracker
    
    def get_results_display(self):
        return self.results_display


# Create main app function with collapsible right column
def create_app():
    # Disclaimer at bottom - smaller text, centered across full window
    disclaimer = pn.pane.Markdown(
        "**📈 Stock Analysis AI**  \n*Powered by Ollama (Gemma3) + LangChain ReAct + LSTM Ensemble + Panel*",
        styles={'text-align': 'center', 'font-size': '14px', 'color': '#666666'},
        margin=(10, 0),
        width_policy='max',
        sizing_mode='stretch_width'
    )
    
    # Create components
    left_sidebar = StockSidebar()
    chat_interface = ChatInterface()
    right_sidebar = RightSidebar()
    
    # Connect components
    left_sidebar.set_chat_interface(chat_interface)
    chat_interface.set_external_components(
        right_sidebar.get_progress_tracker(),
        right_sidebar.get_results_display()
    )
    
    # Create collapsible right panel
    right_panel_visible = pn.widgets.Checkbox(value=True, name="", width=0, height=0, visible=False)
    
    # Toggle button for right panel
    toggle_right_btn = pn.widgets.Button(
        name="Hide Panel ▶",
        button_type="light",
        width=120,
        height=32,
        styles={
            'background': '#F8F9FA',
            'border': '1px solid #E5E5EA', 
            'border-radius': '8px',
            'font-size': '12px',
            'font-weight': '600'
        }
    )
    
    # Right column content
    right_column = pn.Column(
        right_sidebar.get_layout(),
        margin=(10, 10),
        sizing_mode='stretch_width',
        visible=True
    )
    
    def toggle_right_panel(event):
        """Toggle right panel visibility"""
        current_visible = right_column.visible
        right_column.visible = not current_visible
        
        if right_column.visible:
            toggle_right_btn.name = "Hide Panel ▶"
        else:
            toggle_right_btn.name = "◀ Show Panel"
    
    toggle_right_btn.on_click(toggle_right_panel)
    
    # Add toggle button to chat interface header
    chat_interface.set_header_with_toggle(toggle_right_btn)
    
    # Main layout with collapsible right column
    main_row = pn.Row(
        # Left column - Chat and Input + Controls (always visible)
        pn.Column(
            # Chat interface with toggle button in header
            chat_interface.get_layout(),
            pn.Spacer(height=15),
            # Bottom: All three controls horizontally
            left_sidebar.get_bottom_controls(),
            margin=(10, 10),
            sizing_mode='stretch_width'
        ),
        pn.Spacer(width=20),
        
        # Right column - Analysis Status & Results (collapsible)
        right_column,
        sizing_mode='stretch_width'
    )
    
    # Clean main app layout
    app = pn.Column(
        main_row,
        pn.Spacer(height=20),
        # Disclaimer at bottom center
        disclaimer,
        sizing_mode='stretch_width'
    )
    
    return app

# Create and serve the application
logger.info("Creating Stock Analysis Panel Application")
app = create_app()
app.servable()

# Desktop app functionality
def create_desktop_app():
    """Create native macOS desktop app using PyQt6"""
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMenuBar, QMenu
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtCore import QUrl, QTimer
        from PyQt6.QtGui import QAction, QIcon
        import webbrowser
        import subprocess
        import socket
        import time
    except ImportError as e:
        print(f"❌ PyQt6 import failed: {e}")
        print("Install with: pip install PyQt6 PyQt6-WebEngine")
        return False
    
    class StockAnalysisDesktopApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.panel_server_process = None
            self.init_ui()
            self.start_panel_server()
        
        def init_ui(self):
            """Initialize the native macOS UI"""
            # Window setup
            self.setWindowTitle("Stock Analysis AI")
            self.setGeometry(100, 100, 1600, 1000)
            
            # Set custom icon
            self.set_app_icon()
            
            # Create web view
            self.web_view = QWebEngineView()
            self.setCentralWidget(self.web_view)
            
            # Create native macOS menu bar
            self.create_menu_bar()
            
            # Style for native macOS look
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                }
            """)
        
        def create_menu_bar(self):
            """Create native macOS menu bar"""
            menubar = self.menuBar()
            
            # File menu
            file_menu = menubar.addMenu('File')
            
            new_analysis = QAction('New Analysis', self)
            new_analysis.setShortcut('Cmd+N')
            new_analysis.triggered.connect(self.new_analysis)
            file_menu.addAction(new_analysis)
            
            file_menu.addSeparator()
            
            quit_action = QAction('Quit', self)
            quit_action.setShortcut('Cmd+Q')
            quit_action.triggered.connect(self.close)
            file_menu.addAction(quit_action)
            
            # View menu
            view_menu = menubar.addMenu('View')
            
            reload_action = QAction('Reload', self)
            reload_action.setShortcut('Cmd+R')
            reload_action.triggered.connect(self.reload_page)
            view_menu.addAction(reload_action)
            
            fullscreen_action = QAction('Enter Full Screen', self)
            fullscreen_action.setShortcut('Cmd+Ctrl+F')
            fullscreen_action.triggered.connect(self.toggle_fullscreen)
            view_menu.addAction(fullscreen_action)
            
            # Help menu
            help_menu = menubar.addMenu('Help')
            
            about_action = QAction('About Stock Analysis AI', self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
        
        def set_app_icon(self):
            """Set custom app icon for taskbar and dock"""
            from PyQt6.QtGui import QPixmap, QPainter, QBrush, QFont
            from PyQt6.QtCore import Qt
            
            # Create a custom icon programmatically
            pixmap = QPixmap(128, 128)
            pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Background circle with gradient effect
            from PyQt6.QtGui import QRadialGradient, QColor
            gradient = QRadialGradient(64, 64, 60)
            gradient.setColorAt(0, QColor("#007AFF"))
            gradient.setColorAt(0.7, QColor("#0051D5"))
            gradient.setColorAt(1, QColor("#003DA3"))
            
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(4, 4, 120, 120)
            
            # Draw stock chart line
            from PyQt6.QtGui import QPen
            pen = QPen(QColor("white"), 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            
            # Stock chart path (upward trend)
            from PyQt6.QtCore import QPoint
            points = [
                QPoint(25, 85),
                QPoint(35, 75),
                QPoint(45, 80),
                QPoint(55, 60),
                QPoint(65, 50),
                QPoint(75, 45),
                QPoint(85, 35),
                QPoint(95, 40),
                QPoint(105, 25)
            ]
            
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
            
            # Add stock symbol text
            painter.setPen(QPen(QColor("white"), 1))
            font = QFont("SF Pro Display", 16, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(35, 105, "AI")
            
            painter.end()
            
            # Set the icon
            icon = QIcon(pixmap)
            self.setWindowIcon(icon)
            
            # Also set for the application
            from PyQt6.QtWidgets import QApplication
            QApplication.instance().setWindowIcon(icon)
        
        def start_panel_server(self):
            """Start Panel server in background"""
            import threading
            import time
            
            def run_server():
                try:
                    # Check if port is available
                    port = 5007
                    if self.is_port_in_use(port):
                        print(f"Port {port} already in use, trying to connect...")
                    else:
                        print(f"Starting Panel server on port {port}...")
                        # Start server without showing browser
                        pn.serve(app, port=port, show=False, autoreload=False, 
                               allow_websocket_origin=[f"localhost:{port}"])
                    
                except Exception as e:
                    print(f"Error starting Panel server: {e}")
            
            # Start server in background thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait for server to start, then load page
            QTimer.singleShot(2000, self.load_app)  # Wait 2 seconds
        
        def is_port_in_use(self, port):
            """Check if port is already in use"""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        
        def load_app(self):
            """Load the Panel app in web view"""
            url = "http://localhost:5007"
            print(f"Loading app from {url}")
            
            # Inject JavaScript to suppress Plotly errors in desktop app
            js_code = """
            (function() {
                const originalError = console.error;
                console.error = function() {
                    const message = Array.from(arguments).join(' ');
                    if (!message.includes('Resize must be passed a displayed plot div element')) {
                        originalError.apply(console, arguments);
                    }
                };
                
                window.addEventListener('unhandledrejection', function(e) {
                    if (e.reason && e.reason.message && 
                        e.reason.message.includes('Resize must be passed a displayed plot div element')) {
                        e.preventDefault();
                    }
                });
            })();
            """
            
            # Execute JS after page loads
            def inject_error_handler():
                self.web_view.page().runJavaScript(js_code)
            
            # Connect to loadFinished signal
            self.web_view.loadFinished.connect(lambda: inject_error_handler())
            
            self.web_view.load(QUrl(url))
        
        def new_analysis(self):
            """Handle new analysis menu action"""
            self.web_view.reload()
        
        def reload_page(self):
            """Reload the current page"""
            self.web_view.reload()
        
        def toggle_fullscreen(self):
            """Toggle fullscreen mode"""
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        
        def show_about(self):
            """Show about dialog"""
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.about(self, "About Stock Analysis AI", 
                            "Stock Analysis AI\n\n"
                            "A comprehensive stock analysis system powered by:\n"
                            "• Ollama (Gemma3)\n"
                            "• LangChain ReAct agents\n"
                            "• LSTM ensemble neural networks\n"
                            "• Panel web framework\n\n"
                            "Version 1.0")
        
        def closeEvent(self, event):
            """Handle application close"""
            print("Closing Stock Analysis AI...")
            if self.panel_server_process:
                self.panel_server_process.terminate()
            event.accept()
    
    # Create and run the desktop app
    import sys
    app_qt = QApplication(sys.argv)
    app_qt.setApplicationName("Stock Analysis AI")
    app_qt.setOrganizationName("Stock Analysis AI")
    app_qt.setApplicationDisplayName("Stock Analysis AI")
    
    # Create main window
    main_window = StockAnalysisDesktopApp()
    main_window.show()
    
    print("🚀 Stock Analysis AI desktop app started!")
    print("💡 Use Cmd+Q to quit, Cmd+R to reload")
    
    return app_qt.exec()

# For Panel serve command
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Analysis AI')
    parser.add_argument('--desktop', action='store_true', 
                       help='Launch as native macOS desktop app')
    parser.add_argument('--port', type=int, default=5007,
                       help='Port to run the server on (default: 5007)')
    
    args = parser.parse_args()
    
    if args.desktop:
        # Launch desktop app
        print("🖥️  Launching native macOS desktop app...")
        create_desktop_app()
    else:
        # Launch web version
        print("🌐 Launching web version...")
        pn.serve(app, port=args.port, show=True, autoreload=True)