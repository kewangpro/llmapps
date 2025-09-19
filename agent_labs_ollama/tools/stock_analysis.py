#!/usr/bin/env python3
"""
Stock Analysis Tool
Retrieves and analyzes stock data from Yahoo Finance
"""

import json
import sys
import os
import logging
import warnings
from typing import Dict, Any, List
from datetime import datetime, timedelta
import statistics
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.offline as pyo

# Suppress warnings and yfinance verbose output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

logger = logging.getLogger(__name__)

# Suppress yfinance error messages and info messages
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('yfinance.utils').setLevel(logging.CRITICAL)
logging.getLogger('yfinance.base').setLevel(logging.CRITICAL)

# Suppress all yfinance console output
import sys
from contextlib import contextmanager

@contextmanager
def suppress_yfinance_output():
    """Suppress all yfinance output to stdout/stderr"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

try:
    from curl_cffi import requests
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
    CURL_CFFI_AVAILABLE = True
except ImportError as e:
    YFINANCE_AVAILABLE = False
    CURL_CFFI_AVAILABLE = False

def generate_stock_chart(hist: 'pd.DataFrame', symbol: str, company_name: str) -> Dict[str, Any]:
    """Generate interactive stock price chart using Plotly"""
    try:
        # Prepare data for chart
        dates = hist.index.tolist()

        # Handle MultiIndex columns
        if isinstance(hist.columns, pd.MultiIndex):
            # Extract data from MultiIndex format
            close_prices = hist['Close'].iloc[:, 0] if len(hist['Close'].shape) > 1 else hist['Close']
            high_prices = hist['High'].iloc[:, 0] if len(hist['High'].shape) > 1 else hist['High']
            low_prices = hist['Low'].iloc[:, 0] if len(hist['Low'].shape) > 1 else hist['Low']
            open_prices = hist['Open'].iloc[:, 0] if len(hist['Open'].shape) > 1 else hist['Open']
            volumes = hist['Volume'].iloc[:, 0] if len(hist['Volume'].shape) > 1 else hist['Volume']
        else:
            close_prices = hist['Close']
            high_prices = hist['High']
            low_prices = hist['Low']
            open_prices = hist['Open']
            volumes = hist['Volume']

        # Create the main price chart
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name=f'{symbol} Price',
            increasing_line_color='#00C851',
            decreasing_line_color='#FF4444'
        ))

        # Add moving averages if we have enough data
        if len(close_prices) >= 20:
            ma_20 = close_prices.rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_20,
                mode='lines',
                name='20-day MA',
                line=dict(color='orange', width=2)
            ))

        if len(close_prices) >= 50:
            ma_50 = close_prices.rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_50,
                mode='lines',
                name='50-day MA',
                line=dict(color='blue', width=2)
            ))

        # Update layout
        fig.update_layout(
            title=f'{company_name} ({symbol}) - Stock Price Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        # Generate HTML chart with inline Plotly.js (self-contained)
        chart_html = pyo.plot(fig, output_type='div', include_plotlyjs='inline')

        # Extract chart data for frontend
        chart_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'prices': close_prices.tolist(),
            'symbol': symbol,
            'company_name': company_name,
            'chart_html': chart_html
        }

        return {
            'success': True,
            'chart_data': chart_data
        }

    except Exception as e:
        logger.error(f"Failed to generate chart: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def analyze_stock(symbol: str, period: str = "1y", analysis_type: str = "comprehensive", interval: str = "1d") -> Dict[str, Any]:
    """
    Analyze stock performance using Yahoo Finance data

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period for analysis ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        analysis_type: Type of analysis ('basic', 'comprehensive', 'technical')

    Returns:
        Dictionary with analysis results
    """
    try:
        if not YFINANCE_AVAILABLE:
            return {
                "tool": "stock_analysis",
                "success": False,
                "error": "yfinance library not available. Install with: pip install yfinance curl_cffi"
            }

        if not CURL_CFFI_AVAILABLE:
            return {
                "tool": "stock_analysis",
                "success": False,
                "error": "curl_cffi library not available. Install with: pip install curl_cffi"
            }

        if not symbol:
            return {
                "tool": "stock_analysis",
                "success": False,
                "error": "Stock symbol is required"
            }

        # Create curl_cffi session to bypass Yahoo Finance blocking - exact working pattern
        session = requests.Session(impersonate="chrome")

        # Get historical data using exact working pattern
        try:
            logger.info(f"Fetching data for {symbol.upper()} with period={period}")

            # Use exact working pattern: yf.download(tickers, session=session) - suppress stdout
            with suppress_yfinance_output():
                tickers = symbol.upper()
                hist = yf.download(tickers, period=period, session=session, progress=False)
            logger.debug(f"Initial fetch result - empty: {hist.empty}, shape: {hist.shape if not hist.empty else 'N/A'}")

            if hist.empty:
                # Try fallback periods like the working code
                logger.warning(f"No data with period {period}, trying fallback periods for {symbol}")
                for fallback_period in ["6mo", "3mo", "1mo"]:
                    logger.debug(f"Trying fallback period {fallback_period}")
                    with suppress_yfinance_output():
                        tickers = symbol.upper()
                        hist = yf.download(tickers, period=fallback_period, session=session, progress=False)
                    logger.debug(f"Fallback {fallback_period} result - empty: {hist.empty}")
                    if not hist.empty:
                        logger.info(f"Success with fallback period {fallback_period}, shape: {hist.shape}")
                        break

                if hist.empty:
                    return {
                        "tool": "stock_analysis",
                        "success": False,
                        "error": f"No data found for symbol: {symbol}. Tried periods: {period}, 6mo, 3mo, 1mo. Try common symbols like AAPL, GOOGL, TSLA."
                    }
        except Exception as e:
            logger.error(f"Exception during data fetch for {symbol}: {e}")
            return {
                "tool": "stock_analysis",
                "success": False,
                "error": f"Failed to fetch data for {symbol}: {str(e)}"
            }

        # Get stock info using Ticker with session
        try:
            with suppress_yfinance_output():
                stock = yf.Ticker(symbol.upper(), session=session)
                info = stock.info
        except Exception:
            info = {"symbol": symbol.upper()}

        # Generate chart data
        company_name = info.get("longName", symbol.upper())
        chart_result = generate_stock_chart(hist, symbol.upper(), company_name)

        # Perform analysis
        result = {
            "tool": "stock_analysis",
            "success": True,
            "symbol": symbol.upper(),
            "company_name": company_name,
            "period": period,
            "analysis_type": analysis_type,
            "data_points": len(hist),
            "timestamp": datetime.now().isoformat()
        }

        # Add chart data if successful
        if chart_result.get("success"):
            result["chart_data"] = chart_result["chart_data"]

        if analysis_type == "basic":
            result.update(perform_basic_analysis(hist, info))
        elif analysis_type == "technical":
            result.update(perform_technical_analysis(hist, info))
        else:  # comprehensive
            result.update(perform_comprehensive_analysis(hist, info))

        return result

    except Exception as e:
        return {
            "tool": "stock_analysis",
            "success": False,
            "error": str(e)
        }

def calculate_volatility_ratio(close_col) -> float:
    """Calculate volatility ratio handling MultiIndex columns"""
    std_val = close_col.std()
    mean_val = close_col.mean()

    # Handle MultiIndex format
    if isinstance(std_val, pd.Series):
        std_val = std_val.iloc[0]
    if isinstance(mean_val, pd.Series):
        mean_val = mean_val.iloc[0]

    return std_val / mean_val

def perform_basic_analysis(hist: 'pd.DataFrame', info: dict) -> Dict[str, Any]:
    """Perform basic stock analysis"""
    if hist.empty:
        return {"error": "No historical data available"}

    # Handle MultiIndex columns (new yfinance format) - extract scalar values
    close_col = hist['Close']
    if isinstance(close_col.iloc[-1], pd.Series):
        current_price = close_col.iloc[-1].iloc[0]
        start_price = close_col.iloc[0].iloc[0]
    else:
        current_price = close_col.iloc[-1]
        start_price = close_col.iloc[0]
    price_change = current_price - start_price
    price_change_pct = (price_change / start_price) * 100

    # Handle MultiIndex format for aggregated values
    high_col = hist['High']
    low_col = hist['Low']
    volume_col = hist['Volume']

    high_52w = high_col.max()
    low_52w = low_col.min()
    avg_volume = volume_col.mean()

    if isinstance(high_52w, pd.Series):
        high_52w = high_52w.iloc[0]
    if isinstance(low_52w, pd.Series):
        low_52w = low_52w.iloc[0]
    if isinstance(avg_volume, pd.Series):
        avg_volume = avg_volume.iloc[0]

    return {
        "basic_metrics": {
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "price_change_percentage": round(price_change_pct, 2),
            "period_high": round(high_52w, 2),
            "period_low": round(low_52w, 2),
            "average_volume": int(avg_volume),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry")
        },
        "performance_summary": {
            "trend": "bullish" if price_change_pct > 0 else "bearish",
            "volatility": "high" if calculate_volatility_ratio(close_col) > 0.02 else "low",
            "message": f"Stock is {'up' if price_change_pct > 0 else 'down'} {abs(price_change_pct):.2f}% over the period"
        }
    }

def perform_technical_analysis(hist: 'pd.DataFrame', info: dict) -> Dict[str, Any]:
    """Perform technical analysis with indicators"""
    if hist.empty:
        return {"error": "No historical data available"}

    # Basic metrics
    basic = perform_basic_analysis(hist, info)

    # Technical indicators
    close_prices = hist['Close']

    # Moving averages - handle MultiIndex
    ma_10 = None
    ma_20 = None
    ma_50 = None

    if len(close_prices) >= 10:
        ma_val = close_prices.rolling(window=10).mean().iloc[-1]
        ma_10 = ma_val.iloc[0] if isinstance(ma_val, pd.Series) else ma_val

    if len(close_prices) >= 20:
        ma_val = close_prices.rolling(window=20).mean().iloc[-1]
        ma_20 = ma_val.iloc[0] if isinstance(ma_val, pd.Series) else ma_val

    if len(close_prices) >= 50:
        ma_val = close_prices.rolling(window=50).mean().iloc[-1]
        ma_50 = ma_val.iloc[0] if isinstance(ma_val, pd.Series) else ma_val

    # RSI (simplified)
    if len(close_prices) >= 14:
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        if not rsi_series.empty:
            rsi_val = rsi_series.iloc[-1]
            rsi = rsi_val.iloc[0] if isinstance(rsi_val, pd.Series) else rsi_val
        else:
            rsi = None
    else:
        rsi = None

    # Bollinger Bands
    bb_period = min(20, len(close_prices))
    if bb_period >= 2:
        bb_ma = close_prices.rolling(window=bb_period).mean()
        bb_std = close_prices.rolling(window=bb_period).std()
        bb_upper_val = (bb_ma + 2 * bb_std).iloc[-1]
        bb_lower_val = (bb_ma - 2 * bb_std).iloc[-1]

        bb_upper = bb_upper_val.iloc[0] if isinstance(bb_upper_val, pd.Series) else bb_upper_val
        bb_lower = bb_lower_val.iloc[0] if isinstance(bb_lower_val, pd.Series) else bb_lower_val
    else:
        bb_upper = bb_lower = None

    # Support and resistance levels
    recent_data = close_prices.tail(50) if len(close_prices) >= 50 else close_prices
    resistance_val = recent_data.max()
    support_val = recent_data.min()

    resistance = resistance_val.iloc[0] if isinstance(resistance_val, pd.Series) else resistance_val
    support = support_val.iloc[0] if isinstance(support_val, pd.Series) else support_val

    technical_indicators = {
        "moving_averages": {
            "ma_10": round(ma_10, 2) if ma_10 else None,
            "ma_20": round(ma_20, 2) if ma_20 else None,
            "ma_50": round(ma_50, 2) if ma_50 else None
        },
        "momentum": {
            "rsi": round(rsi, 2) if rsi else None,
            "rsi_signal": "oversold" if rsi and rsi < 30 else "overbought" if rsi and rsi > 70 else "neutral" if rsi else None
        },
        "bollinger_bands": {
            "upper": round(bb_upper, 2) if bb_upper else None,
            "lower": round(bb_lower, 2) if bb_lower else None
        },
        "support_resistance": {
            "resistance": round(resistance, 2),
            "support": round(support, 2)
        }
    }

    return {
        **basic,
        "technical_indicators": technical_indicators,
        "trading_signals": generate_trading_signals(close_prices.iloc[-1].iloc[0] if isinstance(close_prices.iloc[-1], pd.Series) else close_prices.iloc[-1], ma_20, rsi, bb_upper, bb_lower)
    }

def perform_comprehensive_analysis(hist: 'pd.DataFrame', info: dict) -> Dict[str, Any]:
    """Perform comprehensive stock analysis"""
    if hist.empty:
        return {"error": "No historical data available"}

    # Get technical analysis
    technical = perform_technical_analysis(hist, info)

    # Additional comprehensive metrics
    close_prices = hist['Close']
    volumes = hist['Volume']

    # Volatility analysis
    daily_returns = close_prices.pct_change().dropna()
    volatility_val = daily_returns.std() * (252 ** 0.5)  # Annualized volatility
    volatility = volatility_val.iloc[0] if isinstance(volatility_val, pd.Series) else volatility_val

    # Risk metrics
    if not daily_returns.empty:
        max_drawdown = calculate_max_drawdown(close_prices)
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    else:
        max_drawdown = sharpe_ratio = 0

    # Volume analysis
    avg_volume_val = volumes.mean()
    recent_volume_val = volumes.tail(5).mean()

    avg_volume = avg_volume_val.iloc[0] if isinstance(avg_volume_val, pd.Series) else avg_volume_val
    recent_volume = recent_volume_val.iloc[0] if isinstance(recent_volume_val, pd.Series) else recent_volume_val

    volume_trend = "increasing" if recent_volume > avg_volume * 1.1 else "decreasing" if recent_volume < avg_volume * 0.9 else "stable"

    # Price patterns
    price_patterns = analyze_price_patterns(close_prices)

    comprehensive_metrics = {
        "risk_metrics": {
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "beta": info.get("beta"),
            "risk_level": "high" if volatility > 0.3 else "medium" if volatility > 0.15 else "low"
        },
        "volume_analysis": {
            "average_volume": int(avg_volume),
            "recent_volume": int(recent_volume),
            "volume_trend": volume_trend,
            "volume_ratio": round(recent_volume / avg_volume, 2)
        },
        "price_patterns": price_patterns,
        "fundamental_data": {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue")
        }
    }

    return {
        **technical,
        "comprehensive_metrics": comprehensive_metrics,
        "investment_summary": generate_investment_summary(technical, comprehensive_metrics)
    }

def calculate_max_drawdown(prices: 'pd.Series') -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    min_drawdown = drawdown.min()

    # Handle MultiIndex format
    if isinstance(min_drawdown, pd.Series):
        min_drawdown = min_drawdown.iloc[0]

    return min_drawdown

def calculate_sharpe_ratio(returns: 'pd.Series', risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0

    std_val = returns.std()
    mean_val = returns.mean()

    # Handle MultiIndex format
    if isinstance(std_val, pd.Series):
        std_val = std_val.iloc[0]
    if isinstance(mean_val, pd.Series):
        mean_val = mean_val.iloc[0]

    if std_val == 0:
        return 0

    excess_returns = mean_val * 252 - risk_free_rate
    return excess_returns / (std_val * (252 ** 0.5))

def analyze_price_patterns(prices: 'pd.Series') -> Dict[str, Any]:
    """Analyze price patterns and trends"""
    if len(prices) < 10:
        return {"trend": "insufficient_data"}

    # Get scalar values for comparison
    latest_price = prices.iloc[-1]
    if isinstance(latest_price, pd.Series):
        latest_price = latest_price.iloc[0]

    price_10_ago = prices.iloc[-10]
    if isinstance(price_10_ago, pd.Series):
        price_10_ago = price_10_ago.iloc[0]

    # Recent trend (last 10 days)
    recent_trend = "bullish" if latest_price > price_10_ago else "bearish"

    # Medium-term trend (last 30 days if available)
    if len(prices) >= 30:
        price_30_ago = prices.iloc[-30]
        if isinstance(price_30_ago, pd.Series):
            price_30_ago = price_30_ago.iloc[0]
        medium_trend = "bullish" if latest_price > price_30_ago else "bearish"
    else:
        medium_trend = "unknown"

    # Price momentum
    price_5_ago = prices.iloc[-min(5, len(prices))]
    if isinstance(price_5_ago, pd.Series):
        price_5_ago = price_5_ago.iloc[0]

    momentum = (latest_price / price_5_ago - 1) * 100

    return {
        "recent_trend": recent_trend,
        "medium_trend": medium_trend,
        "momentum": round(momentum, 2),
        "trend_strength": "strong" if abs(momentum) > 5 else "weak"
    }

def generate_trading_signals(current_price: float, ma_20: float, rsi: float, bb_upper: float, bb_lower: float) -> Dict[str, str]:
    """Generate trading signals based on technical indicators"""
    signals = []

    if ma_20:
        if current_price > ma_20:
            signals.append("Price above MA20 - bullish signal")
        else:
            signals.append("Price below MA20 - bearish signal")

    if rsi:
        if rsi < 30:
            signals.append("RSI oversold - potential buy signal")
        elif rsi > 70:
            signals.append("RSI overbought - potential sell signal")

    if bb_upper and bb_lower:
        if current_price > bb_upper:
            signals.append("Price above Bollinger upper band - overbought")
        elif current_price < bb_lower:
            signals.append("Price below Bollinger lower band - oversold")

    return {
        "signals": signals,
        "overall_sentiment": "bullish" if any("bullish" in s or "buy" in s for s in signals) else "bearish" if any("bearish" in s or "sell" in s for s in signals) else "neutral"
    }

def generate_investment_summary(technical: dict, comprehensive: dict) -> Dict[str, str]:
    """Generate investment summary and recommendations"""
    risk_level = comprehensive["risk_metrics"]["risk_level"]
    trend = technical["performance_summary"]["trend"]
    volatility = comprehensive["risk_metrics"]["volatility"]

    if trend == "bullish" and risk_level == "low":
        recommendation = "BUY - Strong upward trend with low risk"
    elif trend == "bullish" and risk_level == "medium":
        recommendation = "HOLD/BUY - Positive trend but moderate risk"
    elif trend == "bearish" and risk_level == "high":
        recommendation = "SELL - Downward trend with high risk"
    elif volatility > 30:
        recommendation = "CAUTION - High volatility, consider smaller position"
    else:
        recommendation = "HOLD - Mixed signals, monitor closely"

    return {
        "recommendation": recommendation,
        "risk_assessment": f"{risk_level.title()} risk with {volatility:.1f}% volatility",
        "key_insight": f"Stock shows {trend} trend with {comprehensive['volume_analysis']['volume_trend']} volume"
    }

def main():
    """CLI interface for the stock analysis tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: stock_analysis.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        symbol = args.get("symbol", "")
        period = args.get("period", "1y")
        analysis_type = args.get("analysis_type", "comprehensive")

        if not symbol:
            print(json.dumps({"error": "symbol is required"}))
            sys.exit(1)

        result = analyze_stock(symbol, period, analysis_type)
        print(json.dumps(result, indent=2, default=str))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()