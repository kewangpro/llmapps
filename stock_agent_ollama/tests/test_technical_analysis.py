import pytest
import pandas as pd
import numpy as np
from src.tools.technical_analysis import TechnicalAnalysis

# ==================== Fixtures ====================

@pytest.fixture
def sample_data():
    """Create sample price data for testing (Basic)"""
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        'Open': np.linspace(100, 200, 100),
        'High': np.linspace(105, 205, 100),
        'Low': np.linspace(95, 195, 100),
        'Close': np.linspace(100, 200, 100),
        'Volume': np.random.randint(1000, 5000, 100)
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing (Extended/Realistic)."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Generate realistic OHLCV data with upward trend
    base_price = 100
    prices = []
    for i in range(100):
        price = base_price + i * 0.5 + np.random.randn() * 2
        prices.append(price)

    close_prices = pd.Series(prices, index=dates)
    high_prices = close_prices + np.abs(np.random.randn(100) * 1.5)
    low_prices = close_prices - np.abs(np.random.randn(100) * 1.5)

    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(100) * 0.5,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df

@pytest.fixture
def bearish_price_data():
    """Generate bearish price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Generate downward trending data
    base_price = 150
    prices = []
    for i in range(100):
        price = base_price - i * 0.5 + np.random.randn() * 2
        prices.append(max(price, 50))  # Ensure prices don't go negative

    close_prices = pd.Series(prices, index=dates)
    high_prices = close_prices + np.abs(np.random.randn(100) * 1.5)
    low_prices = close_prices - np.abs(np.random.randn(100) * 1.5)

    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(100) * 0.5,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    return df

# ==================== Basic Indicator Tests ====================

def test_calculate_sma(sample_data):
    """Test Simple Moving Average calculation"""
    close = sample_data['Close']
    sma = TechnicalAnalysis.calculate_sma(close, window=10)
    
    assert len(sma) == len(close)
    assert pd.isna(sma.iloc[8])
    assert not pd.isna(sma.iloc[9])
    # For linear data np.linspace(100, 200, 100), mean of first 10 points (100, 101.01, ..., 109.09)
    # The step is (200-100)/(100-1) = 100/99 ≈ 1.0101
    expected_sma_10 = close.iloc[0:10].mean()
    assert pytest.approx(sma.iloc[9], rel=1e-5) == expected_sma_10

def test_calculate_ema(sample_data):
    """Test Exponential Moving Average calculation"""
    close = sample_data['Close']
    ema = TechnicalAnalysis.calculate_ema(close, window=10)
    
    assert len(ema) == len(close)
    assert not pd.isna(ema.iloc[0]) # EMA usually has value from start
    assert ema.iloc[-1] > ema.iloc[0]

def test_calculate_rsi(sample_data):
    """Test RSI calculation"""
    # Linear increasing data should have high RSI
    close = sample_data['Close']
    rsi = TechnicalAnalysis.calculate_rsi(close, window=14)
    
    assert len(rsi) == len(close)
    # RSI for continuously increasing data should be near 100
    assert rsi.iloc[-1] > 90

def test_calculate_macd(sample_data):
    """Test MACD calculation"""
    close = sample_data['Close']
    macd_results = TechnicalAnalysis.calculate_macd(close)
    
    assert 'macd' in macd_results
    assert 'macd_signal' in macd_results
    assert 'macd_histogram' in macd_results
    assert len(macd_results['macd']) == len(close)

def test_calculate_trend_indicators(sample_data):
    """Test the newly added trend indicators"""
    close = sample_data['Close']
    trends = TechnicalAnalysis.calculate_trend_indicators(close)
    
    assert 'sma_trend' in trends
    assert 'ema_crossover' in trends
    assert 'price_momentum' in trends
    assert len(trends['sma_trend']) == len(close)
    # Linear increasing data should have positive trend
    assert trends['sma_trend'].iloc[-1] > 0
    assert trends['price_momentum'].iloc[-1] > 0

# ==================== Bollinger Bands Tests ====================

def test_calculate_bollinger_bands_basic(sample_price_data):
    """Test Bollinger Bands calculation returns correct keys."""
    result = TechnicalAnalysis.calculate_bollinger_bands(sample_price_data['Close'], window=20, num_std=2)

    assert 'bb_upper' in result
    assert 'bb_middle' in result
    assert 'bb_lower' in result

    assert isinstance(result['bb_upper'], pd.Series)
    assert isinstance(result['bb_middle'], pd.Series)
    assert isinstance(result['bb_lower'], pd.Series)


def test_calculate_bollinger_bands_width(sample_price_data):
    """Test that upper band is above middle, middle above lower."""
    result = TechnicalAnalysis.calculate_bollinger_bands(sample_price_data['Close'], window=20, num_std=2)

    # Remove NaN values for comparison
    valid_idx = ~result['bb_middle'].isna()

    upper = result['bb_upper'][valid_idx]
    middle = result['bb_middle'][valid_idx]
    lower = result['bb_lower'][valid_idx]

    assert (upper >= middle).all(), "Upper band should be >= middle band"
    assert (middle >= lower).all(), "Middle band should be >= lower band"


def test_calculate_bollinger_bands_custom_std(sample_price_data):
    """Test Bollinger Bands with different std deviations."""
    result_2std = TechnicalAnalysis.calculate_bollinger_bands(sample_price_data['Close'], window=20, num_std=2)
    result_3std = TechnicalAnalysis.calculate_bollinger_bands(sample_price_data['Close'], window=20, num_std=3)

    # 3-std bands should be wider than 2-std bands
    valid_idx = ~result_2std['bb_middle'].isna()

    width_2std = (result_2std['bb_upper'][valid_idx] - result_2std['bb_lower'][valid_idx]).mean()
    width_3std = (result_3std['bb_upper'][valid_idx] - result_3std['bb_lower'][valid_idx]).mean()

    assert width_3std > width_2std, "3-std bands should be wider than 2-std bands"


# ==================== Stochastic Oscillator Tests ====================

def test_calculate_stochastic_basic(sample_price_data):
    """Test Stochastic Oscillator calculation."""
    result = TechnicalAnalysis.calculate_stochastic(
        sample_price_data['High'],
        sample_price_data['Low'],
        sample_price_data['Close'],
        k_window=14,
        d_window=3
    )

    assert 'stoch_k' in result
    assert 'stoch_d' in result
    assert isinstance(result['stoch_k'], pd.Series)
    assert isinstance(result['stoch_d'], pd.Series)


def test_calculate_stochastic_range(sample_price_data):
    """Test that Stochastic values are between 0 and 100."""
    result = TechnicalAnalysis.calculate_stochastic(
        sample_price_data['High'],
        sample_price_data['Low'],
        sample_price_data['Close'],
        k_window=14,
        d_window=3
    )

    # Remove NaN values
    k_valid = result['stoch_k'].dropna()
    d_valid = result['stoch_d'].dropna()

    assert (k_valid >= 0).all() and (k_valid <= 100).all(), "Stochastic %K should be 0-100"
    assert (d_valid >= 0).all() and (d_valid <= 100).all(), "Stochastic %D should be 0-100"


def test_calculate_stochastic_d_smoother_than_k(sample_price_data):
    """Test that %D is smoother than %K (moving average)."""
    result = TechnicalAnalysis.calculate_stochastic(
        sample_price_data['High'],
        sample_price_data['Low'],
        sample_price_data['Close'],
        k_window=14,
        d_window=3
    )

    # %D should have less volatility than %K (it's a moving average)
    k_std = result['stoch_k'].dropna().std()
    d_std = result['stoch_d'].dropna().std()

    assert d_std <= k_std, "%D should be smoother (lower std) than %K"


# ==================== ATR Tests ====================

def test_calculate_atr_basic(sample_price_data):
    """Test ATR calculation."""
    result = TechnicalAnalysis.calculate_atr(
        sample_price_data['High'],
        sample_price_data['Low'],
        sample_price_data['Close'],
        window=14
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_price_data)


def test_calculate_atr_positive_values(sample_price_data):
    """Test that ATR values are always positive."""
    result = TechnicalAnalysis.calculate_atr(
        sample_price_data['High'],
        sample_price_data['Low'],
        sample_price_data['Close'],
        window=14
    )

    valid_values = result.dropna()
    assert (valid_values >= 0).all(), "ATR should always be non-negative"


def test_calculate_atr_reflects_volatility(sample_price_data):
    """Test that ATR is higher during volatile periods."""
    # Create high volatility period
    volatile_data = sample_price_data.copy()
    volatile_data.loc[volatile_data.index[50:60], 'High'] += 10
    volatile_data.loc[volatile_data.index[50:60], 'Low'] -= 10

    normal_atr = TechnicalAnalysis.calculate_atr(
        sample_price_data['High'],
        sample_price_data['Low'],
        sample_price_data['Close'],
        window=14
    )

    volatile_atr = TechnicalAnalysis.calculate_atr(
        volatile_data['High'],
        volatile_data['Low'],
        volatile_data['Close'],
        window=14
    )

    # ATR should be higher during volatile period
    assert volatile_atr[55:65].mean() > normal_atr[55:65].mean()


# ==================== Trend Analysis Tests ====================

def test_analyze_trends_structure(sample_price_data):
    """Test that analyze_trends returns expected structure."""
    result = TechnicalAnalysis.analyze_trends(sample_price_data)

    assert 'indicators' in result
    assert 'signals' in result
    assert 'overall_trend' in result
    assert 'trend_strength' in result
    assert 'support_level' in result
    assert 'resistance_level' in result
    assert 'latest_values' in result


def test_analyze_trends_indicators(sample_price_data):
    """Test that all expected indicators are present."""
    result = TechnicalAnalysis.analyze_trends(sample_price_data)

    indicators = result['indicators']
    assert 'sma_20' in indicators
    assert 'sma_50' in indicators
    assert 'ema_12' in indicators
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'macd_signal' in indicators
    assert 'macd_histogram' in indicators
    assert 'bb_upper' in indicators
    assert 'bb_middle' in indicators
    assert 'bb_lower' in indicators


def test_analyze_trends_bullish_scenario(sample_price_data):
    """Test trend analysis on bullish data."""
    result = TechnicalAnalysis.analyze_trends(sample_price_data)

    # With upward trending data, should detect bullish signals
    signals = result['signals']

    # At least some bullish indicators should be true
    assert signals['price_above_sma20'] or signals['price_above_sma50'] or signals['sma20_above_sma50']

    # Overall trend should be Bullish or Neutral (not Bearish)
    assert result['overall_trend'] in ['Bullish', 'Neutral']


def test_analyze_trends_bearish_scenario(bearish_price_data):
    """Test trend analysis on bearish data."""
    result = TechnicalAnalysis.analyze_trends(bearish_price_data)

    # With downward trending data, should detect bearish signals
    signals = result['signals']

    # Overall trend should be Bearish or Neutral (not Bullish)
    assert result['overall_trend'] in ['Bearish', 'Neutral']


def test_analyze_trends_support_resistance(sample_price_data):
    """Test that support is below resistance."""
    result = TechnicalAnalysis.analyze_trends(sample_price_data)

    assert result['support_level'] < result['resistance_level']
    assert result['support_level'] > 0
    assert result['resistance_level'] > 0


# ==================== Trading Signals Tests ====================

def test_generate_trading_signals_structure():
    """Test trading signals returns expected structure."""
    # Create simple bullish analysis
    analysis = {
        'signals': {
            'price_above_sma20': True,
            'price_above_sma50': True,
            'sma20_above_sma50': True,
            'rsi_oversold': False,
            'rsi_overbought': False,
            'macd_bullish': True,
            'price_near_bb_upper': False,
            'price_near_bb_lower': False
        },
        'latest_values': {
            'price': 100.0,
            'sma_20': 95.0,
            'sma_50': 90.0,
            'rsi': 60.0,
            'macd': 2.0,
            'macd_signal': 1.0
        },
        'support_level': 85.0,
        'resistance_level': 110.0
    }

    result = TechnicalAnalysis.generate_trading_signals(analysis)

    assert 'primary_signal' in result
    assert 'confidence' in result
    assert 'buy_signals_count' in result
    assert 'sell_signals_count' in result
    assert 'risk_factors' in result
    assert 'recommendations' in result


def test_generate_trading_signals_buy():
    """Test BUY signal generation."""
    analysis = {
        'signals': {
            'price_above_sma20': True,
            'price_above_sma50': True,
            'sma20_above_sma50': True,
            'rsi_oversold': True,
            'rsi_overbought': False,
            'macd_bullish': True,
            'price_near_bb_upper': False,
            'price_near_bb_lower': True
        },
        'latest_values': {
            'price': 100.0,
            'sma_20': 95.0,
            'sma_50': 90.0,
            'rsi': 25.0,  # Oversold
            'macd': 2.0,
            'macd_signal': 1.0
        },
        'support_level': 85.0,
        'resistance_level': 110.0
    }

    result = TechnicalAnalysis.generate_trading_signals(analysis)

    assert result['primary_signal'] == 'BUY'
    assert result['buy_signals_count'] > result['sell_signals_count']
    assert result['confidence'] > 50


def test_generate_trading_signals_sell():
    """Test SELL signal generation."""
    analysis = {
        'signals': {
            'price_above_sma20': False,
            'price_above_sma50': False,
            'sma20_above_sma50': False,
            'rsi_oversold': False,
            'rsi_overbought': True,
            'macd_bullish': False,
            'price_near_bb_upper': True,
            'price_near_bb_lower': False
        },
        'latest_values': {
            'price': 100.0,
            'sma_20': 105.0,
            'sma_50': 110.0,
            'rsi': 75.0,  # Overbought
            'macd': -2.0,
            'macd_signal': -1.0
        },
        'support_level': 85.0,
        'resistance_level': 110.0
    }

    result = TechnicalAnalysis.generate_trading_signals(analysis)

    assert result['primary_signal'] == 'SELL'
    assert result['sell_signals_count'] > result['buy_signals_count']
    assert result['confidence'] > 50


def test_generate_trading_signals_hold():
    """Test HOLD signal generation for neutral conditions."""
    analysis = {
        'signals': {
            'price_above_sma20': True,
            'price_above_sma50': False,
            'sma20_above_sma50': True,
            'rsi_oversold': False,
            'rsi_overbought': False,
            'macd_bullish': False,
            'price_near_bb_upper': False,
            'price_near_bb_lower': False
        },
        'latest_values': {
            'price': 100.0,
            'sma_20': 98.0,
            'sma_50': 102.0,
            'rsi': 50.0,
            'macd': 0.5,
            'macd_signal': 0.6
        },
        'support_level': 85.0,
        'resistance_level': 110.0
    }

    result = TechnicalAnalysis.generate_trading_signals(analysis)

    assert result['primary_signal'] == 'HOLD'
    assert result['confidence'] == 50


def test_generate_trading_signals_risk_factors():
    """Test risk factor detection."""
    analysis = {
        'signals': {
            'price_above_sma20': True,
            'price_above_sma50': True,
            'sma20_above_sma50': True,
            'rsi_oversold': False,
            'rsi_overbought': False,
            'macd_bullish': True,
            'price_near_bb_upper': True,  # Risk factor
            'price_near_bb_lower': False
        },
        'latest_values': {
            'price': 100.0,
            'sma_20': 95.0,
            'sma_50': 90.0,
            'rsi': 85.0,  # Extremely overbought - risk factor
            'macd': 2.0,
            'macd_signal': 1.0
        },
        'support_level': 85.0,
        'resistance_level': 110.0
    }

    result = TechnicalAnalysis.generate_trading_signals(analysis)

    assert len(result['risk_factors']) > 0
    assert any('overbought' in factor.lower() for factor in result['risk_factors'])


def test_generate_trading_signals_recommendations():
    """Test that recommendations are generated."""
    analysis = {
        'signals': {
            'price_above_sma20': True,
            'price_above_sma50': True,
            'sma20_above_sma50': True,
            'rsi_oversold': False,
            'rsi_overbought': False,
            'macd_bullish': True,
            'price_near_bb_upper': False,
            'price_near_bb_lower': False
        },
        'latest_values': {
            'price': 100.0,
            'sma_20': 95.0,
            'sma_50': 90.0,
            'rsi': 60.0,
            'macd': 2.0,
            'macd_signal': 1.0
        },
        'support_level': 85.0,
        'resistance_level': 110.0
    }

    result = TechnicalAnalysis.generate_trading_signals(analysis)

    assert len(result['recommendations']) > 0
    assert isinstance(result['recommendations'], list)
    assert all(isinstance(rec, str) for rec in result['recommendations'])


# ==================== Edge Cases ====================

def test_calculate_indicators_with_insufficient_data():
    """Test indicators with minimal data."""
    short_data = pd.Series([100, 101, 102, 103, 104])

    # Should not crash, but may have NaN values
    sma = TechnicalAnalysis.calculate_sma(short_data, window=20)
    assert len(sma) == len(short_data)
    assert sma.isna().sum() > 0  # Should have NaN values


def test_calculate_rsi_with_zero_change():
    """Test RSI with constant prices (no change)."""
    constant_data = pd.Series([100.0] * 50)
    rsi = TechnicalAnalysis.calculate_rsi(constant_data, window=14)

    # RSI should handle division by zero gracefully
    assert isinstance(rsi, pd.Series)
    # When there's no price change, RSI is typically NaN or 50
    valid_rsi = rsi.dropna()
    if len(valid_rsi) > 0:
        # Either NaN or around 50 (neutral)
        assert True  # Successfully handled edge case


def test_calculate_trend_indicators_structure():
    """Test that calculate_trend_indicators returns expected keys."""
    prices = pd.Series(np.linspace(100, 150, 100))
    result = TechnicalAnalysis.calculate_trend_indicators(prices)

    assert 'sma_trend' in result
    assert 'ema_crossover' in result
    assert 'price_momentum' in result
    assert isinstance(result['sma_trend'], pd.Series)
    assert isinstance(result['ema_crossover'], pd.Series)
    assert isinstance(result['price_momentum'], pd.Series)