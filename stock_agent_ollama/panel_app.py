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

# Configure Panel with extensions
pn.extension('plotly')

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
                self.charts_section
            )),
            active=[0],  # Expanded by default
            width=560,
            visible=False
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
            
            # Current price card
            current_price = stock_data.get('current_price')
            if current_price:
                price_card = pn.pane.HTML(f"""
                <div style='background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;'>
                    <h4>💰 Current Price</h4>
                    <h3>${current_price:.2f}</h3>
                </div>
                """, width=150)
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
            
            # Color based on positive/negative change
            change_color = '#e8f5e8' if percentage_change >= 0 else '#ffebee'
            change_icon = '📈' if percentage_change >= 0 else '📉'
            change_sign = '+' if percentage_change > 0 else ''
            
            change_card = pn.pane.HTML(f"""
            <div style='background: {change_color}; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;'>
                <h4>{change_icon} Percentage Change</h4>
                <h5>{change_sign}{percentage_change:.2f}%</h5>
            </div>
            """, width=150)
            metrics.append(change_card)
            
            # Prediction cards
            if predictions and hasattr(predictions, 'iloc') and len(predictions) > 0:
                try:
                    latest_pred = predictions.iloc[-1]
                    pred_price = latest_pred.get('Predicted_Price')
                    if pred_price:
                        pred_card = pn.pane.HTML(f"""
                        <div style='background: #f1f8e9; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;'>
                            <h4>🔮 Next Day Prediction</h4>
                            <h3>${pred_price:.2f}</h3>
                        </div>
                        """, width=150)
                        metrics.append(pred_card)
                    
                    # Trend card
                    trend = latest_pred.get('Trend', 'N/A')
                    trend_card = pn.pane.HTML(f"""
                    <div style='background: #fff3e0; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;'>
                        <h4>📊 Trend</h4>
                        <h4>{trend}</h4>
                    </div>
                    """, width=150)
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
                chart_pane = pn.pane.Plotly(charts['main'], width=540, height=400)
                self.charts_section.append(chart_pane)
                logger.info(f"Added main chart for {symbol}")
            
            # Display volume chart
            if 'volume' in charts and charts['volume'] is not None:
                self.charts_section.append(pn.pane.Markdown("### 📊 Volume Analysis"))
                volume_pane = pn.pane.Plotly(charts['volume'], width=540, height=300)
                self.charts_section.append(volume_pane)
                logger.info(f"Added volume chart for {symbol}")
            else:
                # Show message if volume data is not available
                self.charts_section.append(pn.pane.Markdown("### 📊 Volume Analysis"))
                self.charts_section.append(pn.pane.Markdown("Volume data not available for this stock."))
            
            # Display trend chart
            if 'trend' in charts:
                self.charts_section.append(pn.pane.Markdown("### 📉 Trend Analysis"))
                trend_pane = pn.pane.Plotly(charts['trend'], width=540, height=300)
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
    
    def create_progress_components(self):
        """Create progress tracking components"""
        self.progress_header = pn.pane.Markdown("### 🔄 Analysis Progress")
        
        # Main progress bar
        self.main_progress = pn.indicators.Progress(
            name="Analysis Steps",
            value=0,
            width=500,
            visible=False,
            bar_color="info"
        )
        
        # Step indicator and details
        self.step_indicator = pn.pane.Markdown("Ready to start analysis...")
        self.step_details = pn.pane.Markdown("")
        
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
        
        self.create_layout()
    
    def create_layout(self):
        """Create progress layout"""
        self.layout = pn.Column(
            self.progress_header,
            self.main_progress,
            self.step_indicator,
            self.step_details,
            pn.Spacer(height=10),
            self.training_container,
            visible=False
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
            
            # Step names and descriptions
            step_names = {
                1: "📊 Fetching Historical Data",
                2: "🧠 LSTM Neural Network Training", 
                3: "📈 Creating Visualizations"
            }
            
            step_details = {
                1: "Downloading stock data from yfinance...",
                2: "Training ensemble models for price prediction...",
                3: "Generating interactive charts and analysis..."
            }
            
            step_name = step_names.get(step, "🔄 Processing")
            step_detail = step_details.get(step, "Working on your analysis...")
            
            self.step_indicator.object = f"**{step_name}** ({step}/{total})"
            self.step_details.object = f"💡 {step_detail}"
            
            # Show training progress when LSTM starts
            if step == 2 and not self.training_started:
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
        self.main_progress.value = 100
        self.step_indicator.object = f"✅ **{message}**"
        self.step_details.object = "All steps completed successfully!"
        
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
            placeholder="e.g., 'Analyze AAPL stock and predict its price for the next 30 days'",
            height=80,
            width=250
        )
        
        self.send_btn = pn.widgets.Button(
            name="Send",
            button_type="primary",
            width=250
        )
        
        # Divider
        self.divider1 = pn.pane.HTML("<hr>")
        
        # Quick Analysis section
        self.quick_analysis_header = pn.pane.Markdown("## 🔍 Quick Analysis")
        
        self.quick_symbol = pn.widgets.TextInput(
            name="Stock Symbol",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            width=250
        )
        
        # Analysis parameters
        self.past_data_period = pn.widgets.Select(
            name="Past Data",
            options=["2y", "5y"],
            value="2y",
            width=120
        )
        
        self.future_prediction = pn.widgets.Select(
            name="Prediction", 
            options=["30 days", "1 week"],
            value="30 days",
            width=120
        )
        
        self.params_row = pn.Row(
            self.past_data_period,
            self.future_prediction
        )
        
        self.quick_analysis_btn = pn.widgets.Button(
            name="🔍 Quick Analysis",
            button_type="primary",
            width=250
        )
        
        # Divider
        self.divider2 = pn.pane.HTML("<hr>")
        
        # Analysis Templates section
        self.templates_header = pn.pane.Markdown("## 📊 Analysis Templates")
        
        self.template_options = {
            "Basic Analysis": "basic_analysis",
            "Trend Analysis": "trend_analysis", 
            "Risk Assessment": "risk_assessment"
        }
        
        self.selected_template = pn.widgets.Select(
            name="Choose template:",
            options=list(self.template_options.keys()),
            value="Basic Analysis",
            width=250
        )
        
        self.template_symbol = pn.widgets.TextInput(
            name="Symbol for template",
            placeholder="e.g., AAPL",
            width=250
        )
        
        self.template_btn = pn.widgets.Button(
            name="📈 Use Template",
            button_type="primary",
            width=250
        )
        
        # Bind events
        self.bind_events()
        self.create_layout()
    
    def bind_events(self):
        """Bind button events"""
        self.send_btn.on_click(self.on_send_message)
        self.quick_analysis_btn.on_click(self.on_quick_analysis)
        self.template_btn.on_click(self.on_template_analysis)
    
    def on_send_message(self, event):
        """Handle send button click"""
        message = self.chat_input.value.strip()
        if message:
            if hasattr(self, 'chat_interface'):
                self.chat_interface.process_user_query(message)
                self.chat_input.value = ""  # Clear input
        else:
            if hasattr(self, 'chat_interface'):
                self.chat_interface.add_system_message("⚠️ Please enter a question first")
    
    def on_quick_analysis(self, event):
        """Handle quick analysis button click"""
        symbol = self.quick_symbol.value.strip() if self.quick_symbol.value else ""
        
        # Debug logging
        logger.info(f"Quick analysis clicked. Symbol input value: '{symbol}'")
        
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
        symbol = self.template_symbol.value.strip()
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
        """Create sidebar layout"""
        self.layout = pn.Column(
            self.chat_header,
            self.chat_input,
            self.send_btn,
            self.divider1,
            self.quick_analysis_header,
            self.quick_symbol,
            self.params_row,
            self.quick_analysis_btn,
            self.divider2,
            self.templates_header,
            self.selected_template,
            self.template_symbol,
            self.template_btn,
            width=280,
            margin=(10, 10)
        )
    
    def get_layout(self):
        return self.layout

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
        
        # Chat history container
        self.chat_history = pn.Column(
            height=500,
            width=500,
            scroll=True,
            auto_scroll_limit=1  # Auto-scroll to bottom when new content is added
        )
        
        # Add initial welcome message
        self.add_assistant_message(
            "👋 Hello! I'm your **AI Stock Analysis Assistant** powered by advanced machine learning.\n\n"
            "**What I can do:**\n"
            "- 📊 Analyze stocks with real-time data\n"
            "- 🧠 Generate LSTM price predictions\n"
            "- 📈 Create interactive charts\n"
            "- 💡 Provide comprehensive analysis\n"
            "- 📉 Assess investment risks\n\n"
            "**Try asking:**\n"
            "- 'Analyze AAPL stock'\n"
            "- 'Predict TSLA trends for 30 days'\n"
            "- 'Compare GOOGL vs MSFT'\n\n"
            "Use the input controls on the left to get started!"
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
                self.external_progress_tracker.update_main_progress(step, total, message)
                
                # Update training progress if this is detailed LSTM training
                if is_detailed_training:
                    self.external_progress_tracker.update_training_progress(message)
                    logger.debug(f"Routing detailed training message: {message}")
                
        except Exception as e:
            logger.error(f"Progress UI update failed: {e}")
    
    def _handle_analysis_result(self, result, query):
        """Handle analysis result"""
        try:
            self.analysis_running = False
            
            if result.get("success", False):
                response = result.get("response", "Analysis completed successfully.")
                self.add_assistant_message(response)
                
                # Complete progress and show results via external components
                if hasattr(self, 'external_progress_tracker') and self.external_progress_tracker:
                    self.external_progress_tracker.complete_progress("Analysis completed!")
                    
                    # Auto-hide progress after delay
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
            styles={'background': '#e3f2fd', 'padding': '10px', 'margin': '5px 0', 'border-radius': '5px'}
        )
        
        self.chat_history.append(user_msg)
        self.messages.append({"role": "user", "content": message, "timestamp": timestamp})
    
    def add_assistant_message(self, message):
        """Add assistant message to chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        assistant_msg = pn.pane.Markdown(
            f"**🤖 Assistant ({timestamp})**\n\n{message}",
            styles={'background': '#f3e5f5', 'padding': '10px', 'margin': '5px 0', 'border-radius': '5px'}
        )
        
        self.chat_history.append(assistant_msg)
        self.messages.append({"role": "assistant", "content": message, "timestamp": timestamp})
    
    def add_system_message(self, message):
        """Add system message to chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        system_msg = pn.pane.Markdown(
            f"**ℹ️ System ({timestamp})**\n\n{message}",
            styles={'background': '#fff3e0', 'padding': '10px', 'margin': '5px 0', 'border-radius': '5px'}
        )
        
        self.chat_history.append(system_msg)
        self.messages.append({"role": "system", "content": message, "timestamp": timestamp})
    
    def create_layout(self):
        """Create chat layout"""
        self.layout = pn.Column(
            pn.pane.Markdown("## 💬 Chat with AI Assistant"),
            self.status,
            self.chat_history,
            width=550
        )
    
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
            width=580,
            margin=(5, 5)
        )
    
    def get_layout(self):
        return self.layout
    
    def get_progress_tracker(self):
        return self.progress_tracker
    
    def get_results_display(self):
        return self.results_display


# Create main app function (similar to step5_results.py structure)
def create_app():
    # Title area
    title = pn.pane.Markdown("# 📈 Stock Analysis AI")
    subtitle = pn.pane.Markdown("*Powered by Ollama (Gemma3) + LangChain ReAct + LSTM Ensemble + Panel*")
    
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
    
    # 3-column layout
    app = pn.Row(
        # Left column - Input controls
        left_sidebar.get_layout(),
        pn.Spacer(width=20),
        
        # Middle column - Chat interface
        pn.Column(
            title,
            subtitle,
            pn.Spacer(height=20),
            chat_interface.get_layout()
        ),
        pn.Spacer(width=10),
        
        # Right column - Progress and results
        right_sidebar.get_layout()
    )
    
    return app

# Create and serve the application
logger.info("Creating Stock Analysis Panel Application")
app = create_app()
app.servable()

# For Panel serve command
if __name__ == "__main__":
    pn.serve(app, port=5007, show=True, autoreload=True)