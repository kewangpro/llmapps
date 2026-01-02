import pandas as pd
import yfinance as yf
import time
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging

from src.config import Config
from src.tools.cache_utils import FileCache

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
        
        # Allow alphanumeric characters, dots, hyphens, and ^ (for indices)
        # This prevents path traversal and other injection attacks
        if not re.match(r'^[\^A-Z0-9.-]+$', symbol):
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
                logger.debug(f"Using cached data for {validated_symbol}")
                
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
                    # Generic case (e.g. loaded via __pandas_json_split__)
                    if isinstance(cached_data, pd.DataFrame):
                        return cached_data
                    
                    # Unknown format, try to create DataFrame directly
                    df = pd.DataFrame(cached_data)
                    if len(df) > 0:
                        end_date = pd.Timestamp.now().normalize()
                        start_date = end_date - pd.Timedelta(days=len(df)-1)
                        df.index = pd.date_range(start=start_date, end=end_date, periods=len(df))
                
                return df
        
        # Fetch from yfinance
        try:
            logger.debug(f"Fetching fresh data for {validated_symbol}")
            ticker = yf.Ticker(validated_symbol)

            # Use start/end dates if provided, otherwise use period
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            else:
                data = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {validated_symbol}")
            
            # Validate data quality
            data = self._validate_and_clean_data(data, validated_symbol)
            
            # Cache the data with index information preserved
            ttl = Config.STOCK_DATA_TTL if interval in ["1m", "5m"] else Config.HISTORICAL_DATA_TTL
            
            # Note: We now rely on SafeJSONEncoder to handle DataFrame caching properly via to_json
            self.cache.set(cache_key, data, ttl=ttl)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {validated_symbol}: {e}")
            raise
    
    def get_real_time_price(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get current price and basic info with short-lived caching"""
        try:
            # Validate and sanitize the symbol
            validated_symbol = self._validate_symbol(symbol)

            # Check cache first (unless force refresh)
            cache_key = f"realtime_{validated_symbol}"
            if not force_refresh:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    logger.debug(f"Using cached real-time data for {validated_symbol}")
                    return cached_data

            # Create a fresh ticker instance to bypass yfinance caching
            logger.debug(f"Fetching fresh real-time data for {validated_symbol}")
            ticker = yf.Ticker(validated_symbol)

            # Use history with 1-minute interval for most recent data
            # This is more reliable than .info which caches heavily
            hist = ticker.history(period='1d', interval='1m', auto_adjust=True)

            if hist.empty:
                # Fallback to daily data
                hist = ticker.history(period='1d', interval='1d', auto_adjust=True)

            if hist.empty:
                 # Fallback to info if history is empty (though unlikely if symbol is valid)
                 info = ticker.info
                 current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                 volume = info.get('volume')
                 timestamp = datetime.now()
            else:
                # Get the most recent row
                latest = hist.iloc[-1]
                current_price = float(latest['Close'])
                volume = int(latest['Volume']) if 'Volume' in latest else 0
                timestamp = latest.name  # Get the timestamp from the index

                # Try to get bid/ask from fast_info
                try:
                    fast_info = ticker.fast_info
                    if hasattr(fast_info, 'last_price'):
                        # Update price if fast_info has newer data
                        if fast_info.last_price and fast_info.last_price > 0:
                            current_price = float(fast_info.last_price)
                except:
                    pass

            # Gather additional info if available
            try:
                info = ticker.info
                market_cap = info.get('marketCap')
                previous_close = info.get('previousClose')
            except:
                market_cap = None
                previous_close = None

            result = {
                'symbol': validated_symbol,
                'current_price': current_price,
                'market_cap': market_cap,
                'volume': volume,
                'previous_close': previous_close,
                'timestamp': timestamp if isinstance(timestamp, str) else timestamp.isoformat()
            }
            
            # Cache the result
            self.cache.set(cache_key, result, ttl=Config.REALTIME_DATA_TTL)
            
            return result

        except Exception as e:
            logger.error(f"Failed to get real-time price for {symbol}: {e}")
            raise

    def is_market_open(self, symbol: str) -> bool:
        """Check if market is currently open (regular hours only)"""
        try:
            validated_symbol = self._validate_symbol(symbol)
            # Create a fresh ticker to get the most up-to-date info
            ticker = yf.Ticker(validated_symbol)
            info = ticker.info
            market_state = info.get('marketState', 'CLOSED').upper()

            # Regular hours only
            return market_state == 'REGULAR'

        except Exception as e:
            logger.warning(f"Could not fetch market state from yfinance: {e}. Falling back to time-based check.")
            # Default to checking trading hours (e.g., 9:30 AM - 4:00 PM ET)
            try:
                from zoneinfo import ZoneInfo
                from datetime import datetime, time

                et_zone = ZoneInfo('US/Eastern')
                now_et = datetime.now(et_zone)

                if now_et.weekday() >= 5:  # Weekend
                    return False

                # Regular market hours: 9:30 AM to 4:00 PM ET
                market_open = time(9, 30) <= now_et.time() < time(16, 0)

                return market_open

            except ImportError:
                 # Fallback for older Python
                now = datetime.now()
                if now.weekday() >= 5:  # Weekend
                    return False
                hour = now.hour
                # This is a rough approximation and assumes server time is close to ET.
                return 9 <= hour < 16

    def fetch_bulk_data(
        self,
        symbols: Union[str, List[str]],
        period: str = "1mo",
        interval: str = "1d",
        group_by: str = 'ticker',
        progress: bool = False,
        threads: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch bulk data for multiple symbols using yf.download.
        Useful for efficiently getting data for watchlists.
        """
        # Generate cache key
        if isinstance(symbols, list):
            tickers_str = " ".join(sorted(symbols))  # Sort to ensure consistent cache key for same symbols
        else:
            tickers_str = symbols

        # Include group_by in cache key since it affects DataFrame structure
        cache_key = f"bulk_{hashlib.md5(tickers_str.encode()).hexdigest()}_{period}_{interval}_{group_by}"

        # Check cache (unless force refresh)
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                if isinstance(cached_data, pd.DataFrame):
                    logger.debug(f"Using cached bulk data for {len(tickers_str.split())} symbols")
                    return cached_data
                # If it's not a DataFrame (e.g. old cache?), try to convert or ignore
                # But with our new SafeJSONEncoder, it should be a DataFrame.
        
        if isinstance(symbols, list):
            tickers = " ".join(symbols)
        else:
            tickers = symbols

        logger.debug(f"Fetching fresh bulk data for {len(tickers.split())} symbols")

        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
            auto_adjust=True,
            progress=progress,
            threads=threads
        )
        
        # Cache data
        self.cache.set(cache_key, data, ttl=Config.BULK_DATA_TTL)
        
        return data

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
    
    def get_stock_info(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive stock information with caching"""
        try:
            # Validate symbol to ensure consistent formatting and safety
            validated_symbol = self._validate_symbol(symbol)

            # Check cache (unless force refresh)
            cache_key = f"info_{validated_symbol}"
            if not force_refresh:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    logger.debug(f"Using cached stock info for {validated_symbol}")
                    return cached_data

            logger.debug(f"Fetching fresh stock info for {validated_symbol}")

            ticker = yf.Ticker(validated_symbol)
            info = ticker.info

            # Ensure we have a current price; fall back to latest close if needed
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price is None:
                # Fallback: use recent history close
                try:
                    data = self.fetch_stock_data(validated_symbol, period="5d", interval="1d")
                    current_price = data['Close'].iloc[-1] if not data.empty else None
                except Exception:
                    current_price = None

            # Attempt to compute fallback financial metrics when missing
            # Compute P/E if trailingPE is missing but trailingEps and price exist
            trailing_eps = info.get('trailingEps')
            trailing_pe = info.get('trailingPE')
            try:
                if trailing_pe is None and trailing_eps not in (None, 0) and current_price not in (None, 0):
                    trailing_pe = float(current_price) / float(trailing_eps)
            except Exception:
                trailing_pe = info.get('trailingPE')

            # Compute market cap from sharesOutstanding * current_price if missing
            market_cap = info.get('marketCap')
            try:
                if market_cap is None and current_price not in (None, 0):
                    shares_outstanding = info.get('sharesOutstanding')
                    if shares_outstanding not in (None, 0):
                        market_cap = float(shares_outstanding) * float(current_price)
            except Exception:
                market_cap = info.get('marketCap')

            # For 52-week high/low and average volume, try to derive from 1y history when missing
            fifty_two_week_high = info.get('fiftyTwoWeekHigh')
            fifty_two_week_low = info.get('fiftyTwoWeekLow')
            average_volume = info.get('averageVolume')
            try:
                need_history = any(v is None for v in (fifty_two_week_high, fifty_two_week_low, average_volume))
                if need_history:
                    hist = self.fetch_stock_data(validated_symbol, period='1y', interval='1d')
                    if not hist.empty:
                        if fifty_two_week_high is None:
                            fifty_two_week_high = float(hist['High'].max())
                        if fifty_two_week_low is None:
                            fifty_two_week_low = float(hist['Low'].min())
                        if average_volume is None:
                            average_volume = float(hist['Volume'].mean())
            except Exception:
                # If history fetch fails, leave values as-is (may be None)
                pass
            
            # Also provide legacy-style top-level keys that the UI expects
            legacy = {
                'trailingPE': trailing_pe,
                'trailingEps': trailing_eps,
                'fiftyTwoWeekHigh': fifty_two_week_high,
                'fiftyTwoWeekLow': fifty_two_week_low,
                'averageVolume': average_volume,
                'marketCap': market_cap,
                'volume': info.get('volume'),
                'previousClose': info.get('previousClose')
            }

            result = {
                'symbol': validated_symbol,
                'name': info.get('longName', info.get('shortName', validated_symbol)),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': market_cap,
                'employees': info.get('fullTimeEmployees'),
                'website': info.get('website'),
                'business_summary': info.get('longBusinessSummary', '')[:500] + '...' if info.get('longBusinessSummary') else '',
                'price_info': {
                    'current_price': current_price,
                    'previous_close': info.get('previousClose'),
                    'day_high': info.get('dayHigh'),
                    'day_low': info.get('dayLow'),
                    'fifty_two_week_high': fifty_two_week_high,
                    'fifty_two_week_low': fifty_two_week_low,
                    'volume': info.get('volume'),
                    'avg_volume': average_volume
                },
                'financial_metrics': {
                    'pe_ratio': trailing_pe,
                    'forward_pe': info.get('forwardPE'),
                    'price_to_book': info.get('priceToBook'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'eps': trailing_eps,
                    'revenue': info.get('totalRevenue')
                }
            }

            # Merge legacy keys into the top-level result for backward compatibility
            result.update(legacy)

            # Cache the result (use longer TTL since company fundamentals rarely change)
            self.cache.set(cache_key, result, ttl=Config.STOCK_INFO_TTL)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get stock info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}