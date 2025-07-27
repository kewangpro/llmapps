import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import streamlit as st
from data_store import get_data_store


class VisualizerInput(BaseModel):
    symbol: str = Field(description="Stock symbol")


class StockVisualizer(BaseTool):
    name = "stock_visualizer"
    description = """Creates interactive visualizations for stock trends and predictions.
    Input should be JSON format: {"symbol": "AAPL"}"""
    args_schema = VisualizerInput
    
    def _run(self, symbol: str) -> Dict[str, Any]:
        # Handle case where agent passes JSON string instead of parsed args
        if isinstance(symbol, str) and symbol.startswith('{'):
            import json
            try:
                parsed = json.loads(symbol)
                if 'symbol' in parsed:
                    symbol = parsed['symbol']
            except json.JSONDecodeError:
                return {"error": f"Invalid symbol format: {symbol}"}
        
        try:
            # Get data from store
            data_store = get_data_store()
            
            import logging
            logger = logging.getLogger(__name__)
            
            # Get historical data
            stock_data = data_store.get_stock_data(symbol)
            if not stock_data:
                return {"error": f"No historical data found for {symbol}. Please fetch data first."}
            
            historical_data = stock_data['data']
            
            # Get LSTM predictions from store
            prediction_key = f"{symbol}_predictions"
            predictions_data = data_store.get_stock_data(prediction_key)
            logger.info(f"Prediction data found: {predictions_data is not None}")
            if predictions_data:
                logger.info(f"Prediction data keys: {list(predictions_data.keys())}")
            
            # Convert historical data to DataFrame
            df_hist = pd.DataFrame(historical_data)
            
            # Debug: Check what columns we have
            logger.info(f"Historical data columns: {df_hist.columns.tolist()}")
            logger.info(f"Historical data shape: {df_hist.shape}")
            
            # Handle different possible date column names
            date_column = None
            for col in ['Date', 'date', 'Datetime', 'datetime']:
                if col in df_hist.columns:
                    date_column = col
                    break
            
            if date_column is None:
                # If no date column found, create one from index if possible
                if hasattr(df_hist, 'index') and len(df_hist) > 0:
                    df_hist = df_hist.reset_index()
                    if 'Date' in df_hist.columns:
                        date_column = 'Date'
                    else:
                        # Create a synthetic date column
                        logger.warning("No date column found, creating synthetic dates")
                        df_hist['Date'] = pd.date_range(start='2023-01-01', periods=len(df_hist), freq='D')
                        date_column = 'Date'
                else:
                    return {"error": "No date information found in historical data"}
            
            df_hist['Date'] = pd.to_datetime(df_hist[date_column])
            
            # Check if we have predictions
            if predictions_data and 'predictions' in predictions_data:
                predictions = predictions_data['predictions']
                future_dates = predictions_data['future_dates']
                trend_analysis = predictions_data.get('trend_analysis', {})
                
                logger.info(f"Found {len(predictions)} predictions for {len(future_dates)} dates")
                
                # Create predictions DataFrame
                df_pred = pd.DataFrame({
                    'Date': pd.to_datetime(future_dates),
                    'Predicted_Close': predictions
                })
                logger.info(f"Prediction DataFrame shape: {df_pred.shape}")
                logger.info(f"Prediction DataFrame columns: {df_pred.columns.tolist()}")
            else:
                # No predictions available yet
                logger.warning("No predictions data available for visualization")
                predictions = []
                future_dates = []
                trend_analysis = {}
                df_pred = pd.DataFrame()
            
            # Create main price chart
            fig_main = go.Figure()
            
            # Add historical prices
            fig_main.add_trace(go.Scatter(
                x=df_hist['Date'],
                y=df_hist['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            # Add predictions if available
            if not df_pred.empty and 'Date' in df_pred.columns and 'Predicted_Close' in df_pred.columns:
                fig_main.add_trace(go.Scatter(
                    x=df_pred['Date'],
                    y=df_pred['Predicted_Close'],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='Date: %{x}<br>Predicted Price: $%{y:.2f}<extra></extra>'
                ))
            
            # Add connection point
            if len(df_hist) > 0 and len(df_pred) > 0:
                fig_main.add_trace(go.Scatter(
                    x=[df_hist['Date'].iloc[-1], df_pred['Date'].iloc[0]],
                    y=[df_hist['Close'].iloc[-1], df_pred['Predicted_Close'].iloc[0]],
                    mode='lines',
                    line=dict(color='orange', width=2),
                    name='Transition',
                    showlegend=False
                ))
            
            fig_main.update_layout(
                title=f'{symbol} Stock Price - Historical vs Predicted',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                template='plotly_white',
                height=600
            )
            
            # Create volume chart if available
            fig_volume = None
            if 'Volume' in df_hist.columns:
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(
                    x=df_hist['Date'],
                    y=df_hist['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))
                fig_volume.update_layout(
                    title=f'{symbol} Trading Volume',
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    template='plotly_white',
                    height=300
                )
            
            # Create trend analysis chart
            fig_trend = go.Figure()
            
            # Recent 30 days trend
            recent_data = df_hist.tail(30)
            fig_trend.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data['Close'],
                mode='lines+markers',
                name='Recent Trend (30 days)',
                line=dict(color='green', width=3),
                marker=dict(size=4)
            ))
            
            # Add trend line
            if len(recent_data) > 1:
                x_numeric = list(range(len(recent_data)))
                z = np.polyfit(x_numeric, recent_data['Close'], 1)
                p = np.poly1d(z)
                
                # Calculate actual trend direction from slope
                slope = z[0]  # First coefficient is the slope
                actual_trend_direction = "Upward" if slope > 0 else "Downward"
                
                fig_trend.add_trace(go.Scatter(
                    x=recent_data['Date'],
                    y=p(x_numeric),
                    mode='lines',
                    name=f'Trend Line ({actual_trend_direction.lower()})',
                    line=dict(color='purple', width=2, dash='dot')
                ))
            
            fig_trend.update_layout(
                title=f'{symbol} Recent Trend Analysis',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                height=400
            )
            
            # Create summary metrics
            current_price = trend_analysis.get('current_price', 0)
            predicted_price = trend_analysis.get('predicted_price_30d', 0)
            price_change = trend_analysis.get('price_change', 0)
            percentage_change = trend_analysis.get('percentage_change', 0)
            
            summary_data = {
                'Metric': ['Current Price', 'Predicted Price (30d)', 'Price Change', 'Percentage Change'],
                'Value': [f'${current_price:.2f}', f'${predicted_price:.2f}', 
                         f'${price_change:.2f}', f'{percentage_change:.2f}%'],
                'Color': ['blue', 'red', 'green' if price_change > 0 else 'red', 
                         'green' if percentage_change > 0 else 'red']
            }
            
            visualization_result = {
                "main_chart": fig_main.to_json(),
                "volume_chart": fig_volume.to_json() if fig_volume else None,
                "trend_chart": fig_trend.to_json(),
                "summary_metrics": summary_data,
                "insights": self._generate_insights(trend_analysis, symbol),
                "chart_objects": {
                    "main": fig_main,
                    "volume": fig_volume,
                    "trend": fig_trend
                }
            }
            
            # Store visualization results in data store for app access
            viz_key = f"{symbol}_visualizations"
            data_store.store_stock_data(viz_key, visualization_result)
            logger.info(f"Stored visualization data for {symbol}")
            
            return {
                "symbol": symbol,
                "status": "success",
                "message": f"Interactive charts and visualizations created for {symbol}",
                "insights_count": len(visualization_result["insights"]),
                "charts_created": ["main_chart", "volume_chart", "trend_chart"]
            }
            
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Error creating visualizations: {str(e)}"}
    
    def _generate_insights(self, trend_analysis: Dict, symbol: str) -> List[str]:
        insights = []
        
        direction = trend_analysis.get('direction', 'Unknown')
        strength = trend_analysis.get('strength_percent', 0)
        percentage_change = trend_analysis.get('percentage_change', 0)
        
        insights.append(f"The model predicts a {direction.lower()} trend for {symbol}")
        
        if abs(percentage_change) > 10:
            insights.append(f"Significant price movement expected: {percentage_change:.1f}%")
        elif abs(percentage_change) > 5:
            insights.append(f"Moderate price movement expected: {percentage_change:.1f}%")
        else:
            insights.append(f"Stable price movement expected: {percentage_change:.1f}%")
        
        if strength > 15:
            insights.append("High volatility detected in recent trading patterns")
        elif strength > 8:
            insights.append("Moderate volatility observed")
        else:
            insights.append("Low volatility, indicating stable trading")
        
        return insights
    
    async def _arun(self, historical_data: List[Dict], predictions: List[float], 
                    future_dates: List[str], symbol: str, trend_analysis: Dict) -> Dict[str, Any]:
        return self._run(historical_data, predictions, future_dates, symbol, trend_analysis)


# Utility functions for Streamlit integration
def display_charts(visualization_data: Dict):
    """Display charts in Streamlit"""
    if "chart_objects" in visualization_data:
        charts = visualization_data["chart_objects"]
        
        # Main chart
        if charts.get("main"):
            st.plotly_chart(charts["main"], use_container_width=True)
        
        # Volume chart
        if charts.get("volume"):
            st.plotly_chart(charts["volume"], use_container_width=True)
        
        # Trend chart
        if charts.get("trend"):
            st.plotly_chart(charts["trend"], use_container_width=True)

def display_summary_metrics(summary_data: Dict):
    """Display summary metrics in Streamlit"""
    if summary_data:
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        
        for i, (metric, value, color) in enumerate(zip(
            summary_data['Metric'], 
            summary_data['Value'], 
            summary_data['Color']
        )):
            with cols[i]:
                st.metric(label=metric, value=value)

import numpy as np