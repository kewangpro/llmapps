#!/usr/bin/env python3
"""
Stock Analysis AI Platform - Setup Verification Script

This script verifies that all components are properly installed and configured.
"""

import sys
import os
import importlib
import subprocess
import traceback
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version"""
    try:
        module = importlib.import_module(import_name or package_name)
        version = getattr(module, '__version__', 'Unknown')
        
        if min_version and hasattr(module, '__version__'):
            from packaging import version as pkg_version
            if pkg_version.parse(module.__version__) < pkg_version.parse(min_version):
                print(f"⚠️ {package_name} version {version} (requires {min_version}+)")
                return False
        
        print(f"✅ {package_name} (v{version})")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️ {package_name} - Warning: {str(e)}")
        return True

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        Path("data"),
        Path("data/cache"),
        Path("data/models"),
        Path("data/logs"),
        Path("src"),
        Path("src/agents"),
        Path("src/tools"),
        Path("src/ui"),
        Path("src/utils")
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory}")
            all_exist = False
    
    return all_exist

def test_stock_fetcher():
    """Test stock data fetching functionality"""
    try:
        from src.tools.stock_fetcher import StockFetcher
        fetcher = StockFetcher()
        
        print("🔍 Testing stock data fetching...")
        # Test with a simple fetch (should be quick)
        data = fetcher.fetch_stock_data("AAPL", period="5d")
        
        if not data.empty and len(data) > 0:
            print(f"✅ Stock data fetch successful ({len(data)} records)")
            return True
        else:
            print("❌ Stock data fetch returned empty data")
            return False
            
    except Exception as e:
        print(f"❌ Stock data fetch failed: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_cache_system():
    """Test caching functionality"""
    try:
        from src.utils.cache_utils import FileCache
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileCache(Path(temp_dir))
            
            # Test cache set/get
            test_data = {"test": "data", "number": 123}
            cache.set("test_key", test_data, ttl=10)
            
            retrieved_data = cache.get("test_key")
            if retrieved_data == test_data:
                print("✅ Cache system working")
                return True
            else:
                print("❌ Cache system data mismatch")
                return False
                
    except Exception as e:
        print(f"❌ Cache system failed: {str(e)}")
        return False

def test_tensorflow():
    """Test TensorFlow installation and basic functionality"""
    try:
        import tensorflow as tf
        
        # Test basic TensorFlow operation
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        z = tf.matmul(x, y)
        
        print(f"✅ TensorFlow working (v{tf.__version__})")
        
        # Check available devices
        devices = tf.config.list_physical_devices()
        print(f"   Available devices: {len(devices)}")
        for device in devices:
            print(f"   - {device.name}: {device.device_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {str(e)}")
        return False

def test_lstm_predictor():
    """Test LSTM predictor functionality"""
    try:
        from src.tools.lstm_predictor import LSTMPredictor
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic stock data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        test_data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        predictor = LSTMPredictor()
        
        # Test data preparation
        X, y, scaler = predictor.prepare_data(test_data)
        
        if len(X) > 50 and len(y) > 50:
            print("✅ LSTM data preparation successful")
            print(f"   Prepared {len(X)} sequences for training")
            return True
        else:
            print(f"⚠️ LSTM data preparation: insufficient data ({len(X)} sequences)")
            return False
            
    except Exception as e:
        print(f"❌ LSTM predictor test failed: {str(e)}")
        return False

def test_technical_analysis():
    """Test technical analysis functionality"""
    try:
        from src.tools.technical_analysis import TechnicalAnalysis
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        test_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Test SMA calculation
        sma = TechnicalAnalysis.calculate_sma(test_data['Close'], 20)
        
        if len(sma.dropna()) > 0:
            print("✅ Technical analysis working")
            return True
        else:
            print("❌ Technical analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Technical analysis test failed: {str(e)}")
        return False

def test_query_processor():
    """Test query processing functionality"""
    try:
        from src.agents.query_processor import QueryProcessor
        
        processor = QueryProcessor()
        
        # Test intent extraction
        intent, entities = processor._extract_intent_and_entities("analyze aapl")
        
        if intent == 'analyze' and 'symbols' in entities and 'AAPL' in entities['symbols']:
            print("✅ Query processor working")
            return True
        else:
            print(f"❌ Query processor failed: intent={intent}, entities={entities}")
            return False
            
    except Exception as e:
        print(f"❌ Query processor test failed: {str(e)}")
        return False

def test_panel_components():
    """Test Panel UI components"""
    try:
        from src.ui.components import StockAnalysisApp
        import panel as pn
        
        # Test creating the app
        app = StockAnalysisApp()
        layout = app.create_layout()
        
        if layout is not None:
            print("✅ Panel UI components working")
            return True
        else:
            print("❌ Panel UI components failed")
            return False
            
    except Exception as e:
        print(f"❌ Panel UI test failed: {str(e)}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 Stock Analysis AI Platform - Setup Verification\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Directories", check_directories),
        ("Core Dependencies", lambda: all([
            check_package("panel", min_version="1.3.0"),
            check_package("pandas", min_version="2.0.0"),
            check_package("numpy", min_version="1.24.0"),
            check_package("requests", min_version="2.31.0"),
            check_package("pydantic", min_version="2.0.0"),
        ])),
        ("Data & Visualization", lambda: all([
            check_package("yfinance", min_version="0.2.0"),
            check_package("plotly", min_version="5.17.0"),
            check_package("bokeh", min_version="3.3.0"),
        ])),
        ("Machine Learning", lambda: all([
            check_package("tensorflow", min_version="2.15.0"),
            check_package("scikit-learn", "sklearn", min_version="1.3.0"),
            check_package("joblib", min_version="1.3.0"),
        ])),
        ("TensorFlow Functionality", test_tensorflow),
        ("Cache System", test_cache_system),
        ("Stock Data Fetching", test_stock_fetcher),
        ("Technical Analysis", test_technical_analysis),
        ("LSTM Predictor", test_lstm_predictor),
        ("Query Processor", test_query_processor),
        ("Panel UI Components", test_panel_components),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} - Unexpected error: {str(e)}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Setup Verification Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your Stock Analysis AI Platform is ready!")
        print("\n🚀 To start the application:")
        print("   python src/main.py")
        print("\n🌐 Then open your browser to: http://localhost:5006")
        return 0
    else:
        print(f"\n⚠️ {total - passed} checks failed. Please review the errors above.")
        print("\n📝 Common solutions:")
        print("   • Run: pip install -r requirements.txt")
        print("   • Check your Python version (3.8+ required)")
        print("   • Ensure all directories exist")
        return 1

if __name__ == "__main__":
    sys.exit(main())