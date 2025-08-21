import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Visualizer:
    """Stock data visualization using Plotly"""
    
    def __init__(self):
        self.default_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_price_chart(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        chart_type: str = "candlestick",
        include_volume: bool = True,
        predictions: Optional[Dict[str, Any]] = None,
        technical_indicators: Optional[Dict[str, pd.Series]] = None
    ) -> go.Figure:
        """Create comprehensive price chart with optional predictions"""
        
        if data.empty:
            return self._create_empty_chart(f"No data available for {symbol}")
        
        # Determine subplot configuration
        rows = 2 if include_volume else 1
        subplot_titles = [f"{symbol} Price Chart"]
        if include_volume:
            subplot_titles.append("Volume")
        
        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=subplot_titles,
            row_heights=[0.7, 0.3] if include_volume else [1.0]
        )
        
        # Add price chart
        if chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol,
                    increasing_line_color=self.default_colors['success'],
                    decreasing_line_color=self.default_colors['danger']
                ),
                row=1, col=1
            )
        else:  # line chart
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f"{symbol} Close",
                    line=dict(color=self.default_colors['primary'], width=2)
                ),
                row=1, col=1
            )
        
        # Add technical indicators
        if technical_indicators:
            for indicator_name, indicator_data in technical_indicators.items():
                if indicator_data is not None and not indicator_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_data.index,
                            y=indicator_data.values,
                            mode='lines',
                            name=indicator_name,
                            line=dict(width=1),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
        
        # Add predictions if provided
        if predictions and predictions.get('predictions'):
            pred_dates = pd.to_datetime(predictions['dates'])
            pred_values = predictions['predictions']
            
            # Convert prediction dates to Python datetime for consistency with pandas 2.0+
            pred_dates_plotly = [d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in pred_dates]
            
            # Main prediction line
            fig.add_trace(
                go.Scatter(
                    x=pred_dates_plotly,
                    y=pred_values,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color=self.default_colors['warning'], width=2, dash='dash'),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Confidence intervals
            if 'confidence_upper' in predictions and 'confidence_lower' in predictions:
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates_plotly,
                        y=predictions['confidence_upper'],
                        mode='lines',
                        name='Upper Confidence',
                        line=dict(color=self.default_colors['warning'], width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates_plotly,
                        y=predictions['confidence_lower'],
                        mode='lines',
                        fill='tonexty',
                        name='Confidence Interval',
                        line=dict(color=self.default_colors['warning'], width=0),
                        fillcolor=f"rgba(255, 127, 14, 0.2)"
                    ),
                    row=1, col=1
                )
        
        # Add volume chart if requested
        if include_volume:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Stock Analysis",
            template="plotly_white",
            showlegend=True,
            hovermode='x unified',
            height=700 if include_volume else 400,
            margin=dict(l=60, r=50, t=100, b=80)
        )
        
        # Remove range slider and selector for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        # Update y-axis labels for each subplot
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if include_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            # Add Date label only to the bottom subplot (volume chart)
            fig.update_xaxes(title_text="Date", row=2, col=1)
        else:
            # Add Date label to the only subplot when no volume chart
            fig.update_xaxes(title_text="Date", row=1, col=1)
        
        return fig
    
    def create_comparison_chart(self, data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create comparison chart for multiple stocks"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (symbol, data) in enumerate(data_dict.items()):
            if not data.empty:
                # Normalize prices to percentage change from first day
                normalized_data = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=normalized_data,
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
        
        fig.update_layout(
            title="Stock Performance Comparison (% Change)",
            xaxis_title="Date",
            yaxis_title="Percentage Change (%)",
            template="plotly_white",
            hovermode='x unified',
            margin=dict(l=60, r=50, t=100, b=80)
        )
        
        return fig
    
    def create_technical_analysis_chart(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        indicators: Dict[str, Any]
    ) -> go.Figure:
        """Create technical analysis chart with multiple indicators"""
        
        # Create subplots for different indicator types
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=[
                f"{symbol} Price & Moving Averages",
                "RSI",
                "MACD"
            ],
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Add moving averages if provided
        if 'sma_20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd_signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='red', width=2)
                ),
                row=3, col=1
            )
            
            if 'macd_histogram' in indicators:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=indicators['macd_histogram'],
                        name='MACD Histogram',
                        marker_color='gray',
                        opacity=0.6
                    ),
                    row=3, col=1
                )
        
        fig.update_layout(
            title=f"{symbol} Technical Analysis",
            template="plotly_white",
            height=900,
            showlegend=True,
            margin=dict(l=60, r=50, t=100, b=80)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        # Add Date label only to the bottom subplot (MACD chart)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def create_prediction_chart(
        self, 
        historical_data: pd.DataFrame,
        predictions: Dict[str, Any],
        symbol: str
    ) -> go.Figure:
        """Create focused prediction visualization"""
        
        fig = go.Figure()
        
        # Show last 60 days of historical data for context
        recent_data = historical_data.tail(60)
        
        # Historical prices
        # Convert historical data index to Python datetime for consistency with pandas 2.0+
        historical_dates_plotly = [d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in recent_data.index]
        
        fig.add_trace(
            go.Scatter(
                x=historical_dates_plotly,
                y=recent_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color=self.default_colors['primary'], width=2)
            )
        )
        
        # Predictions
        pred_dates = pd.to_datetime(predictions['dates'])
        pred_values = predictions['predictions']
        
        # Convert prediction dates to Python datetime for consistency with pandas 2.0+
        pred_dates_plotly = [d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in pred_dates]
        
        # Connect last historical point to first prediction
        # Use the already converted historical dates for consistency
        last_historical_date = historical_dates_plotly[-1]
        first_pred_date = pred_dates_plotly[0]
        
        connection_x = [last_historical_date, first_pred_date]
        connection_y = [recent_data['Close'].iloc[-1], pred_values[0]]
        
        fig.add_trace(
            go.Scatter(
                x=connection_x,
                y=connection_y,
                mode='lines',
                name='Connection',
                line=dict(color=self.default_colors['warning'], width=2, dash='dot'),
                showlegend=False
            )
        )
        
        # Main prediction line
        fig.add_trace(
            go.Scatter(
                x=pred_dates_plotly,
                y=pred_values,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color=self.default_colors['warning'], width=3),
                marker=dict(size=6)
            )
        )
        
        # Confidence intervals
        if 'confidence_upper' in predictions and 'confidence_lower' in predictions:
            fig.add_trace(
                go.Scatter(
                    x=pred_dates_plotly,
                    y=predictions['confidence_upper'],
                    mode='lines',
                    name='Upper Confidence',
                    line=dict(color=self.default_colors['warning'], width=0),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_dates_plotly,
                    y=predictions['confidence_lower'],
                    mode='lines',
                    fill='tonexty',
                    name='Prediction Range',
                    line=dict(color=self.default_colors['warning'], width=0),
                    fillcolor=f"rgba(255, 127, 14, 0.2)"
                )
            )
        
        # Add vertical line to separate historical from predicted using shapes instead of add_vline
        # This avoids the pandas 2.0+ arithmetic issues with add_vline
        last_date_plotly = historical_dates_plotly[-1]
        
        # Get y-axis range for the vertical line
        all_prices = list(recent_data['Close']) + pred_values
        if 'confidence_upper' in predictions and 'confidence_lower' in predictions:
            all_prices.extend(predictions['confidence_upper'])
            all_prices.extend(predictions['confidence_lower'])
        
        y_min, y_max = min(all_prices), max(all_prices)
        y_range = y_max - y_min
        y_margin = y_range * 0.1  # 10% margin
        
        fig.add_shape(
            type="line",
            x0=last_date_plotly, x1=last_date_plotly,
            y0=y_min - y_margin, y1=y_max + y_margin,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Add annotation for the line
        fig.add_annotation(
            x=last_date_plotly,
            y=y_max + y_margin * 0.5,
            text="Today",
            showarrow=False,
            font=dict(color="gray", size=10)
        )
        
        fig.update_layout(
            title=f"{symbol} Price Prediction ({predictions.get('prediction_period_days', 30)} days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            hovermode='x unified',
            margin=dict(l=60, r=50, t=100, b=80)
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig