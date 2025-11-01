import pandas as pd
import yfinance as yf
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from src.config import Config
from src.utils.cache_utils import FileCache

logger = logging.getLogger(__name__)

class StockFetcher:
    """Stock data fetcher with intelligent caching"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Config.CACHE_DIR / "stock_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = FileCache(self.cache_dir)
    
    def _validate_symbol(self, symbol: str) -> str:
        """Validate and sanitize stock symbol to prevent injection attacks"""
        if not symbol:
            raise ValueError("Stock symbol cannot be empty")
        
        # Convert to string and strip whitespace
        symbol = str(symbol).strip().upper()
        
        # Allow only alphanumeric characters, dots, and hyphens (common in stock symbols)
        # This prevents path traversal and other injection attacks
        if not re.match(r'^[A-Z0-9.-]+$', symbol):
            raise ValueError(f"Invalid stock symbol format: {symbol}")
        
        # Limit length to reasonable stock symbol length
        if len(symbol) > 10:
            raise ValueError(f"Stock symbol too long: {symbol}")
        
        return symbol
        
    def fetch_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch stock data with caching.

        Args:
            symbol: Stock ticker symbol
            period: Time period (e.g., "1y", "6mo") - ignored if start_date/end_date provided
            interval: Data interval (e.g., "1d", "1h")
            force_refresh: Force refresh from API
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)

        Returns:
            DataFrame with stock data
        """

        # Validate and sanitize the symbol
        validated_symbol = self._validate_symbol(symbol)

        # Generate cache key using validated symbol
        if start_date and end_date:
            cache_key = f"{validated_symbol}_{start_date}_{end_date}_{interval}"
        else:
            cache_key = f"{validated_symbol}_{period}_{interval}"
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for {validated_symbol}")
                
                # Handle both old format (list) and new format (dict)
                if isinstance(cached_data, list):
                    # Old format: just a list of records, create DataFrame and fix index
                    df = pd.DataFrame(cached_data)
                    # Create a synthetic datetime index since we don't have the original
                    if len(df) > 0:
                        end_date = pd.Timestamp.now().normalize()
                        start_date = end_date - pd.Timedelta(days=len(df)-1)
                        df.index = pd.date_range(start=start_date, end=end_date, periods=len(df))
                elif isinstance(cached_data, dict) and 'data' in cached_data:
                    # New format: dict with data and index
                    df = pd.DataFrame(cached_data['data'])
                    # Restore the datetime index
                    if 'index' in cached_data and cached_data['index']:
                        df.index = pd.to_datetime(cached_data['index'])
                    else:
                        # Fallback to synthetic index
                        end_date = pd.Timestamp.now().normalize()
                        start_date = end_date - pd.Timedelta(days=len(df)-1)
                        df.index = pd.date_range(start=start_date, end=end_date, periods=len(df))
                else:
                    # Unknown format, try to create DataFrame directly
                    df = pd.DataFrame(cached_data)
                    end_date = pd.Timestamp.now().normalize()
                    start_date = end_date - pd.Timedelta(days=len(df)-1)
                    df.index = pd.date_range(start=start_date, end=end_date, periods=len(df))
                
                return df
        
        # Fetch from yfinance
        try:
            logger.info(f"Fetching fresh data for {validated_symbol}")
            ticker = yf.Ticker(validated_symbol)

            # Use start/end dates if provided, otherwise use period
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {validated_symbol}")
            
            # Validate data quality
            data = self._validate_and_clean_data(data, validated_symbol)
            
            # Cache the data with index information preserved
            ttl = Config.STOCK_DATA_TTL if interval in ["1m", "5m"] else Config.HISTORICAL_DATA_TTL
            cached_data = {
                'data': data.to_dict('records'),
                'index': data.index.strftime('%Y-%m-%d').tolist() if isinstance(data.index, pd.DatetimeIndex) else data.index.tolist()
            }
            self.cache.set(cache_key, cached_data, ttl=ttl)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {validated_symbol}: {e}")
            raise
    
    def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price and basic info"""
        try:
            # Validate and sanitize the symbol
            validated_symbol = self._validate_symbol(symbol)
            
            ticker = yf.Ticker(validated_symbol)
            info = ticker.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price is None:
                # Fallback to latest close from history
                data = self.fetch_stock_data(validated_symbol, period="5d", interval="1d")
                current_price = data['Close'].iloc[-1] if not data.empty else None
            
            return {
                'symbol': validated_symbol,
                'current_price': current_price,
                'market_cap': info.get('marketCap'),
                'volume': info.get('volume'),
                'previous_close': info.get('previousClose'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time price for {validated_symbol}: {e}")
            raise
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        results = {}
        for symbol in symbols:
            try:
                # Validation is handled in fetch_stock_data
                validated_symbol = self._validate_symbol(symbol)
                results[validated_symbol] = self.fetch_stock_data(validated_symbol, period)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                # Use validated symbol for the key if possible
                try:
                    validated_symbol = self._validate_symbol(symbol)
                    results[validated_symbol] = pd.DataFrame()
                except:
                    results[symbol] = pd.DataFrame()  # Fallback for invalid symbols
        return results
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean stock data"""
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Remove rows with missing values
        initial_length = len(data)
        data = data.dropna()
        if len(data) < initial_length * 0.9:  # If we lose more than 10% of data
            logger.warning(f"Significant data loss for {symbol}: {initial_length} -> {len(data)} rows")
        
        # Check for obviously wrong values
        if (data['High'] < data['Low']).any():
            logger.warning(f"Data quality issue for {symbol}: High < Low detected")
        
        if (data['Close'] <= 0).any():
            logger.warning(f"Data quality issue for {symbol}: Non-positive close prices")
            data = data[data['Close'] > 0]
        
        # Sort by date to ensure chronological order
        data = data.sort_index()
        
        return data
    
    def get_available_symbols(self) -> List[str]:
        """Get list of popular stock symbols for suggestions"""
        return [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'SPOT',
            'TEAM', 'SNOW', 'PLTR', 'ZM', 'NET', 'CRWD', 'OKTA', 'DDOG',
            'DIS', 'BA', 'GE', 'F', 'GM', 'JPM', 'BAC', 'WFC', 'GS',
            'SPY', 'QQQ', 'IWM', 'VTI', 'BTC-USD', 'ETH-USD'
        ]
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'employees': info.get('fullTimeEmployees'),
                'website': info.get('website'),
                'business_summary': info.get('longBusinessSummary', '')[:500] + '...' if info.get('longBusinessSummary') else '',
                'price_info': {
                    'current_price': info.get('currentPrice'),
                    'previous_close': info.get('previousClose'),
                    'day_high': info.get('dayHigh'),
                    'day_low': info.get('dayLow'),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                    'volume': info.get('volume'),
                    'avg_volume': info.get('averageVolume')
                },
                'financial_metrics': {
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'price_to_book': info.get('priceToBook'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'eps': info.get('trailingEps'),
                    'revenue': info.get('totalRevenue')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get stock info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}