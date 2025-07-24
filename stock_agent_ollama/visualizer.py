import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import streamlit as st


class VisualizerInput(BaseModel):
    historical_data: List[Dict] = Field(description="Historical stock data")
    predictions: List[float] = Field(description="Predicted prices")
    future_dates: List[str] = Field(description="Future prediction dates")
    symbol: str = Field(description="Stock symbol")
    trend_analysis: Dict = Field(description="Trend analysis data")


class StockVisualizer(BaseTool):
    name = "stock_visualizer"
    description = """Creates interactive visualizations for stock trends and predictions.
    Input should be JSON format: {"historical_data": [...], "predictions": [...], "future_dates": [...], "symbol": "AAPL", "trend_analysis": {...}}"""
    args_schema = VisualizerInput
    
    def _run(self, historical_data: List[Dict], predictions: List[float], 
             future_dates: List[str], symbol: str, trend_analysis: Dict) -> Dict[str, Any]:
        try:
            # Convert historical data to DataFrame
            df_hist = pd.DataFrame(historical_data)
            df_hist['Date'] = pd.to_datetime(df_hist['Date'])
            
            # Create predictions DataFrame
            df_pred = pd.DataFrame({
                'Date': pd.to_datetime(future_dates),
                'Predicted_Close': predictions
            })
            
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
            
            # Add predictions
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
                
                fig_trend.add_trace(go.Scatter(
                    x=recent_data['Date'],
                    y=p(x_numeric),
                    mode='lines',
                    name=f'Trend Line ({trend_analysis["direction"]})',
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
            
            return {
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
            
        except Exception as e:
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