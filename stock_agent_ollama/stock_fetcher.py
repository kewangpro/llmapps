import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging
import time
import random

logger = logging.getLogger(__name__)


class StockFetcherInput(BaseModel):
    symbol: str = Field(description="Stock symbol (e.g., AAPL, GOOGL)")
    period: str = Field(default="2y", description="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")


class StockFetcher(BaseTool):
    name = "stock_fetcher"
    description = """Fetches historical stock price data for a given symbol. 
    Input should be JSON format: {"symbol": "AAPL", "period": "2y"}"""
    args_schema = StockFetcherInput
    
    def _run(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        # Handle case where agent passes JSON string instead of parsed args
        if isinstance(symbol, str) and symbol.startswith('{'):
            logger.warning(f"Received JSON string as symbol: {symbol}")
            import json
            try:
                parsed = json.loads(symbol)
                if 'symbol' in parsed:
                    symbol = parsed['symbol']
                    if 'period' in parsed:
                        period = parsed['period']
                    logger.info(f"Parsed symbol: {symbol}, period: {period}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON symbol: {symbol}")
                return {"error": f"Invalid symbol format: {symbol}"}
        
        logger.info(f"Fetching stock data for {symbol} with period {period}")
        
        # Retry logic for rate limiting
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retry attempt {attempt + 1}, waiting {delay:.1f} seconds...")
                    time.sleep(delay)
                
                ticker = yf.Ticker(symbol.upper())
                
                # Get historical data with timeout
                hist = ticker.history(period=period, timeout=30)
                
                if hist.empty:
                    logger.warning(f"No historical data found for symbol {symbol}")
                    return {"error": f"No data found for symbol {symbol}"}
                
                # Get basic info with error handling
                try:
                    info = ticker.info
                except Exception as info_error:
                    logger.warning(f"Could not fetch ticker info: {info_error}")
                    # Use fallback info if ticker.info fails
                    info = {
                        "longName": f"{symbol.upper()} Company",
                        "currentPrice": float(hist['Close'].iloc[-1]) if len(hist) > 0 else None,
                        "marketCap": "N/A",
                        "trailingPE": "N/A"
                    }
                
                # Prepare data for LSTM
                data = {
                    "symbol": symbol.upper(),
                    "company_name": info.get("longName", "Unknown"),
                    "current_price": info.get("currentPrice", hist['Close'].iloc[-1]),
                    "market_cap": info.get("marketCap", "N/A"),
                    "pe_ratio": info.get("trailingPE", "N/A"),
                    "data": hist.reset_index().to_dict('records'),
                    "latest_close": float(hist['Close'].iloc[-1]),
                    "data_range": f"{hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}",
                    "total_records": len(hist)
                }
                
                logger.info(f"Successfully fetched data for {symbol}: {data['total_records']} records from {data['data_range']}")
                return data
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {error_msg}")
                
                # Check if it's a rate limiting error
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        logger.error(f"Rate limiting persists after {max_retries} attempts")
                        return {"error": f"Yahoo Finance rate limiting detected. Please try again in a few minutes. Symbol: {symbol}"}
                
                # Check if it's a symbol not found error
                elif "No data found" in error_msg or "delisted" in error_msg:
                    return {"error": f"No data found for symbol {symbol}. Please verify the symbol is correct."}
                
                # For other errors, retry once more
                elif attempt < max_retries - 1:
                    continue
                else:
                    logger.error(f"All retry attempts failed for {symbol}")
                    return {"error": f"Unable to fetch data for {symbol} after {max_retries} attempts: {error_msg}"}
    
    async def _arun(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        return self._run(symbol, period)


def get_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Utility function to get stock data as DataFrame"""
    ticker = yf.Ticker(symbol.upper())
    return ticker.history(period=period)