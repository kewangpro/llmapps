import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from src.tools.lstm.data_pipeline import prepare_enhanced_data_robust
from src.tools.lstm.prediction_service import LSTMPredictionService
from src.tools.stock_fetcher import StockFetcher

@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    dates = pd.date_range(start="2023-01-01", periods=200)
    data = {
        'Open': np.linspace(100, 200, 200),
        'High': np.linspace(105, 205, 200),
        'Low': np.linspace(95, 195, 200),
        'Close': np.linspace(100, 200, 200),
        'Volume': np.random.randint(1000, 5000, 200).astype(float)
    }
    return pd.DataFrame(data, index=dates)

def test_data_pipeline_horizon(sample_data):
    """Test that data pipeline handles variable horizons correctly"""
    seq_len = 10
    
    # Horizon = 1 (Default)
    X1, y1, _ = prepare_enhanced_data_robust(sample_data, seq_len, horizon=1)
    
    # Horizon = 5
    X5, y5, _ = prepare_enhanced_data_robust(sample_data, seq_len, horizon=5)
    
    # With H=1, target for input [t-seq...t-1] is t
    # With H=5, target for input [t-seq...t-1] is t+4
    
    # Check lengths. 
    # H=1: Last input ends at len-2, target at len-1. Total = len - seq.
    # H=5: Last input ends at len-6, target at len-1. Total = len - seq - horizon + 1.
    
    expected_len_1 = len(sample_data) - seq_len # approx (due to NaN dropping from TA)
    # The TA calculation drops initial rows (e.g. 50 for SMA50).
    # prepare_enhanced_data_robust calls dropna().
    
    # Let's just compare the difference in length
    # len(X5) should be exactly len(X1) - 4
    
    assert len(X1) - len(X5) == 4

def test_stock_fetcher_news():
    """Test get_stock_news with mocked yfinance"""
    fetcher = StockFetcher()
    
    mock_news_item = {
        'content': {
            'title': 'Test News Title',
            'summary': '<p>Test Summary</p>',
            'pubDate': '2023-01-01',
            'canonicalUrl': {'url': 'http://test.com'},
            'provider': {'displayName': 'TestProvider'}
        }
    }
    
    with patch('yfinance.Ticker') as MockTicker:
        mock_instance = MockTicker.return_value
        # Mock the news property
        type(mock_instance).news = PropertyMock(return_value=[mock_news_item])
        
        news = fetcher.get_stock_news('AAPL')
        
        assert len(news) == 1
        assert news[0]['title'] == 'Test News Title'
        assert news[0]['summary'] == 'Test Summary' # HTML stripped? regex in code handles it
        assert news[0]['link'] == 'http://test.com'

def test_prediction_service_sentiment_adjustment(sample_data):
    """Test that sentiment score modifies predictions"""
    # Mock dependencies
    with patch('src.tools.lstm.prediction_service.load_ensemble_with_fallback') as mock_load:
        with patch('src.tools.lstm.prediction_service.diagnose_model_issues'):
            # Setup mock models
            mock_model = MagicMock()
            mock_scaler = MagicMock()
            mock_scaler.fitted = True
            mock_scaler.price_min = 100
            mock_scaler.price_max = 200
            mock_scaler.data_range = 100
            mock_scaler.n_features_in_ = 17 # Enhanced features
            
            # Mock load return
            mock_load.return_value = ([mock_model], mock_scaler)
            
            service = LSTMPredictionService()
            
            # Mock internal prediction methods to return deterministic values
            # Mock _predict_ensemble to return constant 0.5 (scaled) -> 150 (unscaled)
            service._predict_ensemble = MagicMock(return_value=np.array([150.0]))
            
            # Mock multi-step to return list of 150s (new list each time)
            with patch('src.tools.lstm.prediction_service._generate_multi_step_predictions') as mock_multi:
                mock_multi.side_effect = lambda *args, **kwargs: [150.0] * 30
                
                with patch('src.tools.lstm.prediction_service._generate_ensemble_predictions') as mock_ens_pred:
                    mock_ens_pred.return_value = np.array([[150.0]*30]) # for variance
                    
                    with patch('src.tools.lstm.prediction_service.determine_feature_compatibility') as mock_compat:
                        mock_compat.return_value = {'uses_enhanced_features': True, 'feature_count': 17}
                        
                        with patch('src.tools.lstm.prediction_service.prepare_enhanced_data') as mock_prep:
                            # Mock X shape
                            mock_prep.return_value = (np.zeros((10, 10, 17)), None, None)
                            
                            # 1. Predict without sentiment
                            result_base = service.predict('AAPL', sample_data, days=30)
                            base_preds = result_base['predictions']
                            
                            # 2. Predict with positive sentiment
                            result_bullish = service.predict('AAPL', sample_data, days=30, sentiment_score=1.0)
                            bullish_preds = result_bullish['predictions']
                            
                            # 3. Predict with negative sentiment
                            result_bearish = service.predict('AAPL', sample_data, days=30, sentiment_score=-1.0)
                            bearish_preds = result_bearish['predictions']
                            
                            # Assertions
                            # Last day should be adjusted the most
                            # Base is 150.
                            # Bullish: 150 * (1 + 1.0 * 0.05 * 1.0) = 150 * 1.05 = 157.5
                            # Bearish: 150 * (1 - 1.0 * 0.05 * 1.0) = 150 * 0.95 = 142.5
                            
                            assert bullish_preds[-1] > base_preds[-1]
                            assert bearish_preds[-1] < base_preds[-1]
                            
                            # Check approx values
                            assert abs(bullish_preds[-1] - 157.5) < 1.0
                            assert abs(bearish_preds[-1] - 142.5) < 1.0
                            
                            # Check structure
                            assert result_bullish['sentiment_analysis']['score'] == 1.0
