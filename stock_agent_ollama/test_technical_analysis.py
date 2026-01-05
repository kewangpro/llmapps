import pytest
import pandas as pd
import numpy as np
from src.tools.technical_analysis import TechnicalAnalysis

@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        'Open': np.linspace(100, 200, 100),
        'High': np.linspace(105, 205, 100),
        'Low': np.linspace(95, 195, 100),
        'Close': np.linspace(100, 200, 100),
        'Volume': np.random.randint(1000, 5000, 100)
    }
    return pd.DataFrame(data, index=dates)

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
