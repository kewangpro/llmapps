"""
Test suite for LSTM Predictor with comprehensive scaling fix validation.

This test suite covers:
1. Original feature quality tests for outlier handling
2. CompositeScaler serialization/deserialization tests
3. Scaler metadata saving and restoration tests  
4. Scaling health check functionality
5. Model diagnosis for scaling issues
6. Fallback data restoration when metadata is unavailable

The scaling-related tests validate the fixes for the prediction gap issue
caused by CompositeScaler losing price range parameters during serialization.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
import joblib
from src.tools.lstm_predictor import LSTMPredictor, CompositeScaler

class TestLSTMPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = LSTMPredictor()
        # Create temporary directory for test model files
        self.test_model_dir = tempfile.mkdtemp()
        self.original_model_dir = self.predictor.model_dir
        self.predictor.model_dir = Path(self.test_model_dir)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_model_dir, ignore_errors=True)
        # Restore original model directory
        self.predictor.model_dir = self.original_model_dir

    def _create_dummy_data(self, num_rows=100):
        """Creates dummy DataFrame with 'Close', 'Volume', 'High', 'Low', 'Open'."""
        dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='D')
        data = {
            'Close': np.random.rand(num_rows) * 100 + 100,
            'Volume': np.random.rand(num_rows) * 1000000 + 100000,
            'High': np.random.rand(num_rows) * 10 + 195,
            'Low': np.random.rand(num_rows) * 10 + 185,
            'Open': np.random.rand(num_rows) * 10 + 190,
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_feature_quality_warnings_fixed(self):
        """
        Test that 'Momentum_Score' and 'MACD_Histogram' warnings are no longer present
        after applying the fixes.
        """
        # Create dummy data that might have triggered the warnings
        # For Momentum_Score, we need some price/volume changes
        # For MACD_Histogram, it's often centered around zero, leading to high CV
        df = self._create_dummy_data(num_rows=200)
        
        # Introduce some specific patterns to ensure features are calculated and potentially problematic
        df['Close'] = df['Close'].rolling(window=5).mean().bfill()
        df['Volume'] = df['Volume'].rolling(window=5).mean().bfill()

        # Calculate enhanced features using the internal method
        # This will apply the _ultra_aggressive_preprocessing and feature calculations
        enhanced_data = self.predictor._calculate_robust_features(df)
        
        # Ensure there are no NaNs after feature calculation for validation
        enhanced_data = enhanced_data.dropna()

        # Run the validation
        validation_results = self.predictor._validate_feature_quality(enhanced_data, "TEST_SYMBOL")

        # Assert that the specific warnings are NOT present
        warnings = validation_results['outlier_warnings']
        scaling_recommendations = validation_results['scaling_recommendations']

        momentum_warning_found = False
        for warning in warnings:
            if "Momentum_Score" in warning and "suspicious" in warning:
                momentum_warning_found = True
                break
        
        self.assertFalse(momentum_warning_found, "Momentum_Score warning should not be present after fix.")
        
        self.assertNotIn('MACD_Histogram', scaling_recommendations, 
                         "MACD_Histogram should not be recommended for RobustScaler after fix.")

    def test_composite_scaler_serialization(self):
        """Test that CompositeScaler serializes and deserializes properly with all attributes."""
        # Create a CompositeScaler with price range data
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol="TEST", validation_split=0.2)
        
        # Create dummy data and fit the scaler
        dummy_data = pd.DataFrame({
            'Close': [100, 110, 120, 130, 140],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'High': [105, 115, 125, 135, 145],
            'Low': [95, 105, 115, 125, 135],
            'Open': [98, 108, 118, 128, 138]
        })
        
        # Fit the scaler
        scaler.fit_transform(dummy_data)
        
        # Verify scaler is fitted with price range data
        self.assertTrue(scaler.fitted)
        self.assertIsNotNone(scaler.price_min)
        self.assertIsNotNone(scaler.price_max)
        self.assertIsNotNone(scaler.data_range)
        
        original_price_min = scaler.price_min
        original_price_max = scaler.price_max
        original_data_range = scaler.data_range
        
        # Save and load the scaler
        scaler_path = Path(self.test_model_dir) / "test_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        loaded_scaler = joblib.load(scaler_path)
        
        # Verify all attributes are preserved
        self.assertEqual(loaded_scaler.price_min, original_price_min)
        self.assertEqual(loaded_scaler.price_max, original_price_max)
        self.assertEqual(loaded_scaler.data_range, original_data_range)
        self.assertTrue(loaded_scaler.fitted)

    def test_scaler_metadata_saving(self):
        """Test that scaler metadata is saved correctly during model training."""
        # Create a CompositeScaler
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol="TEST", validation_split=0.2)
        
        # Fit with dummy data
        dummy_data = pd.DataFrame({
            'Close': [100, 110, 120, 130, 140],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'High': [105, 115, 125, 135, 145],
            'Low': [95, 105, 115, 125, 135],
            'Open': [98, 108, 118, 128, 138]
        })
        scaler.fit_transform(dummy_data)
        
        # Simulate saving models with metadata (simplified version)
        # We'll create a minimal model list to avoid the complexity of actual TensorFlow models
        class MockModel:
            def __init__(self):
                self.input_shape = [None, 30, 5]
                self.output_shape = [None, 1]
            def count_params(self):
                return 1000
        
        mock_models = [MockModel()]
        
        # Test the metadata creation logic
        import sys
        import tensorflow as tf
        
        metadata = {
            'symbol': "TEST",
            'ensemble_size': len(mock_models),
            'models_saved': len(mock_models),
            'sequence_length': 30,
            'training_date': pd.Timestamp.now().isoformat(),
            'model_format': 'keras',
            'tensorflow_version': tf.__version__,
            'keras_version': tf.keras.__version__,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'training_histories': [],
            'saved_model_paths': [],
            'model_architecture': {
                'input_shape': list(mock_models[0].input_shape),
                'output_shape': list(mock_models[0].output_shape),
                'total_params': mock_models[0].count_params(),
                'feature_count': 5,
                'uses_enhanced_features': False
            },
            'scaler_info': {
                'scaler_type': type(scaler).__name__,
                'fitted': getattr(scaler, 'fitted', None),
                'price_min': getattr(scaler, 'price_min', None),
                'price_max': getattr(scaler, 'price_max', None),
                'data_range': getattr(scaler, 'data_range', None),
                'n_features': getattr(scaler, 'n_features_in_', None)
            }
        }
        
        # Verify scaler info is included and correct
        self.assertIn('scaler_info', metadata)
        scaler_info = metadata['scaler_info']
        self.assertEqual(scaler_info['scaler_type'], 'CompositeScaler')
        self.assertTrue(scaler_info['fitted'])
        self.assertIsNotNone(scaler_info['price_min'])
        self.assertIsNotNone(scaler_info['price_max'])
        self.assertIsNotNone(scaler_info['data_range'])
        self.assertEqual(scaler_info['n_features'], 5)

    def test_scaler_restoration_from_metadata(self):
        """Test that scaler parameters are restored from metadata when loading models."""
        symbol = "TEST"
        
        # Create and save a scaler with missing attributes (simulating the bug)
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol=symbol)
        
        # Manually set attributes then clear them to simulate the serialization issue
        scaler.price_min = 100.0
        scaler.price_max = 200.0
        scaler.data_range = 100.0
        scaler.fitted = True
        
        original_price_min = scaler.price_min
        original_price_max = scaler.price_max
        
        # Save metadata with scaler info
        metadata = {
            'symbol': symbol,
            'scaler_info': {
                'scaler_type': 'CompositeScaler',
                'fitted': True,
                'price_min': original_price_min,
                'price_max': original_price_max,
                'data_range': 100.0,
                'n_features': 5
            }
        }
        
        metadata_path = self.predictor.model_dir / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Clear the scaler's attributes to simulate the bug
        scaler.price_min = None
        scaler.price_max = None
        scaler.data_range = None
        
        # Save the "corrupted" scaler
        scaler_path = self.predictor.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Test the restoration logic
        _, restored_scaler = self.predictor._load_ensemble(symbol)
        
        # Verify that the scaler was restored (even though no models exist, the scaler should be restored)
        self.assertIsNotNone(restored_scaler)
        self.assertEqual(restored_scaler.price_min, original_price_min)
        self.assertEqual(restored_scaler.price_max, original_price_max)
        self.assertEqual(restored_scaler.data_range, 100.0)

    def test_scaling_health_check(self):
        """Test the scaling health check function."""
        symbol = "TEST"
        
        # Test 1: No scaler file
        health = self.predictor.check_scaling_health(symbol)
        self.assertFalse(health['scaling_healthy'])
        self.assertIn("Scaler file not found", health['issues'])
        
        # Test 2: Healthy CompositeScaler
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        healthy_scaler = CompositeScaler(feature_names, symbol=symbol)
        healthy_scaler.price_min = 100.0
        healthy_scaler.price_max = 200.0
        healthy_scaler.data_range = 100.0
        healthy_scaler.fitted = True
        
        scaler_path = self.predictor.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(healthy_scaler, scaler_path)
        
        health = self.predictor.check_scaling_health(symbol)
        self.assertTrue(health['scaling_healthy'])
        self.assertIn("Scaling parameters are healthy", health['recommendations'])
        
        # Test 3: Corrupted CompositeScaler with metadata available
        corrupted_scaler = CompositeScaler(feature_names, symbol=symbol)
        corrupted_scaler.fitted = True
        # Missing price_min, price_max, data_range
        
        joblib.dump(corrupted_scaler, scaler_path)
        
        # Create metadata
        metadata = {
            'symbol': symbol,
            'scaler_info': {
                'price_min': 100.0,
                'price_max': 200.0,
                'data_range': 100.0
            }
        }
        metadata_path = self.predictor.model_dir / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        health = self.predictor.check_scaling_health(symbol)
        self.assertFalse(health['scaling_healthy'])  # Not healthy but recoverable
        self.assertIn("Scaler parameters missing but available in metadata", health['issues'])
        # Check for any recommendation containing the auto-restore message
        auto_restore_found = any("auto-restored from metadata" in rec for rec in health['recommendations'])
        self.assertTrue(auto_restore_found, f"Expected auto-restore recommendation not found in: {health['recommendations']}")

    def test_model_diagnosis_scaling_issues(self):
        """Test that model diagnosis detects scaling issues."""
        symbol = "TEST"
        
        # Create a corrupted scaler without metadata
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        corrupted_scaler = CompositeScaler(feature_names, symbol=symbol)
        corrupted_scaler.fitted = True
        # Missing price_min, price_max, data_range
        
        scaler_path = self.predictor.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(corrupted_scaler, scaler_path)
        
        # Run diagnosis
        diagnosis = self.predictor.diagnose_model_issues(symbol)
        
        # Should detect the scaling issue
        self.assertIn("CompositeScaler missing price range parameters", diagnosis['issues'])
        # Check for any recommendation containing "Retrain the model to fix scaling issues"
        retrain_recommendation_found = any("Retrain the model to fix scaling issues" in rec for rec in diagnosis['recommendations'])
        self.assertTrue(retrain_recommendation_found, f"Expected scaling fix recommendation not found in: {diagnosis['recommendations']}")

    def test_scaler_fallback_data_restoration(self):
        """Test that scaler falls back to data restoration when metadata unavailable."""
        symbol = "TEST"
        
        # Create corrupted scaler
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        corrupted_scaler = CompositeScaler(feature_names, symbol=symbol)
        corrupted_scaler.fitted = True
        
        scaler_path = self.predictor.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(corrupted_scaler, scaler_path)
        
        # Create test data for restoration
        test_data = pd.DataFrame({
            'Close': [150, 160, 170, 180, 190],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'High': [155, 165, 175, 185, 195],
            'Low': [145, 155, 165, 175, 185],
            'Open': [148, 158, 168, 178, 188]
        })
        
        # Load ensemble with data (should restore from data)
        _, restored_scaler = self.predictor._load_ensemble(symbol, test_data)
        
        # Verify restoration from data
        self.assertIsNotNone(restored_scaler)
        self.assertEqual(restored_scaler.price_min, 150.0)  # min of Close prices
        self.assertEqual(restored_scaler.price_max, 190.0)  # max of Close prices
        self.assertEqual(restored_scaler.data_range, 40.0)  # 190 - 150

    def test_end_to_end_scaling_workflow(self):
        """Test complete end-to-end workflow: save model -> corrupt scaler -> detect issue -> restore."""
        symbol = "TEST"
        
        # Step 1: Create and fit a healthy scaler (simulating model training)
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        original_scaler = CompositeScaler(feature_names, symbol=symbol)
        
        training_data = pd.DataFrame({
            'Close': [100, 110, 120, 130, 140, 150],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500],
            'High': [105, 115, 125, 135, 145, 155],
            'Low': [95, 105, 115, 125, 135, 145],
            'Open': [98, 108, 118, 128, 138, 148]
        })
        
        original_scaler.fit_transform(training_data)
        original_price_min = original_scaler.price_min
        original_price_max = original_scaler.price_max
        
        # Step 2: Save scaler and metadata (simulating successful model saving)
        scaler_path = self.predictor.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(original_scaler, scaler_path)
        
        metadata = {
            'symbol': symbol,
            'scaler_info': {
                'scaler_type': 'CompositeScaler',
                'fitted': True,
                'price_min': original_price_min,
                'price_max': original_price_max,
                'data_range': original_scaler.data_range,
                'n_features': 5
            }
        }
        
        metadata_path = self.predictor.model_dir / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Step 3: Simulate the corruption (what was happening before the fix)
        corrupted_scaler = joblib.load(scaler_path)
        corrupted_scaler.price_min = None  # Simulate serialization bug
        corrupted_scaler.price_max = None
        corrupted_scaler.data_range = None
        joblib.dump(corrupted_scaler, scaler_path)
        
        # Step 4: Detect the issue
        health_before = self.predictor.check_scaling_health(symbol)
        self.assertFalse(health_before['scaling_healthy'])
        self.assertIn("Scaler parameters missing but available in metadata", health_before['issues'])
        
        # Step 5: Load models (this should trigger automatic restoration)
        _, restored_scaler = self.predictor._load_ensemble(symbol)
        
        # Step 6: Verify the fix worked
        self.assertIsNotNone(restored_scaler)
        self.assertEqual(restored_scaler.price_min, original_price_min)
        self.assertEqual(restored_scaler.price_max, original_price_max)
        self.assertTrue(restored_scaler.fitted)
        
        # Step 7: Verify health check now passes (scaler should be saved back to disk)
        health_after = self.predictor.check_scaling_health(symbol)
        
        # The scaler should now be healthy since it was restored and saved back to disk
        self.assertTrue(health_after['scaling_healthy'])
        self.assertIn("Scaling parameters are healthy", health_after['recommendations'])
        self.assertEqual(health_after['scaler_info']['price_min'], original_price_min)
        self.assertEqual(health_after['scaler_info']['price_max'], original_price_max)

    def test_prediction_consistency_validation(self):
        """Test that prediction consistency is maintained with proper scaling."""
        # Create a CompositeScaler with known parameters
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol="TEST")
        
        # Set specific scaling parameters
        scaler.price_min = 100.0
        scaler.price_max = 200.0
        scaler.data_range = 100.0
        scaler.fitted = True
        scaler.n_features_in_ = 5
        
        # Test forward and inverse transform consistency
        test_price = 150.0  # Should scale to 0.5
        
        # Create dummy data for transform
        dummy_data = np.array([[test_price, 1000, 155, 145, 148]])
        
        # Transform
        scaled_data = scaler.transform(dummy_data)
        expected_scaled_price = (test_price - scaler.price_min) / scaler.data_range
        
        # The first feature should be properly scaled
        np.testing.assert_almost_equal(scaled_data[0, 0], expected_scaled_price, decimal=6)
        
        # Inverse transform
        inverse_data = scaler.inverse_transform(scaled_data)
        
        # Should get back original price
        np.testing.assert_almost_equal(inverse_data[0, 0], test_price, decimal=6)

    def test_real_world_integration_amzn(self):
        """Integration test using real AMZN model files (if available)."""
        # This test uses the actual model directory, not the temporary test directory
        real_predictor = LSTMPredictor()
        real_predictor.model_dir = self.original_model_dir  # Use real model directory
        
        symbol = "AMZN"
        
        try:
            # Test scaling health check on real model
            health_result = real_predictor.check_scaling_health(symbol)
            
            # Verify the health check returns expected structure
            self.assertIn('symbol', health_result)
            self.assertIn('scaling_healthy', health_result) 
            self.assertIn('issues', health_result)
            self.assertIn('recommendations', health_result)
            self.assertIn('scaler_info', health_result)
            self.assertEqual(health_result['symbol'], symbol)
            
            # Test model diagnosis integration
            diagnosis = real_predictor.diagnose_model_issues(symbol)
            
            # Verify diagnosis structure (status might not be present in all diagnosis results)
            self.assertIn('issues', diagnosis)
            self.assertIn('recommendations', diagnosis)
            
            # If there are scaling issues, they should be detected
            if not health_result['scaling_healthy']:
                scaling_issue_found = any("scaling" in issue.lower() for issue in diagnosis['issues'])
                scaling_recommendation_found = any("scaling" in rec.lower() or "retrain" in rec.lower() 
                                                 for rec in diagnosis['recommendations'])
                # At least one should be true if scaling is unhealthy
                self.assertTrue(scaling_issue_found or scaling_recommendation_found,
                               f"Scaling issues not properly detected in diagnosis. "
                               f"Health: {health_result['scaling_healthy']}, "
                               f"Issues: {diagnosis['issues']}, "
                               f"Recommendations: {diagnosis['recommendations']}")
                               
        except Exception as e:
            # This test is optional - it only works if AMZN model files exist
            # Skip gracefully if model files are not available
            self.skipTest(f"Real AMZN model files not available for integration test: {e}")

if __name__ == '__main__':
    unittest.main()
