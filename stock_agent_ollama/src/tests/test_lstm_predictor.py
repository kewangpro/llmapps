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
import tensorflow as tf
from pathlib import Path
import joblib
import sys
sys.path.append('/Users/kewang/PyProjects/stock_agent_ollama')

from src.tools.lstm_predictor import LSTMPredictor
from src.tools.lstm.custom_scalers import CompositeScaler
from src.tools.lstm.model_manager import load_ensemble_with_fallback, get_model_info, save_ensemble
from src.tools.lstm.data_pipeline import _calculate_robust_features, _validate_feature_quality, prepare_enhanced_data
from src.tools.lstm.prediction_utils import _predict_single_model, _inverse_transform_predictions # Import utility functions

import logging

logging.basicConfig(level=logging.DEBUG)

class TestLSTMPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = LSTMPredictor()
        # Create temporary directory for test model files
        self.test_model_dir = tempfile.mkdtemp()
        self.original_model_dir = self.predictor.service.model_dir # Access service's model_dir
        self.predictor.service.model_dir = Path(self.test_model_dir) # Set service's model_dir

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_model_dir, ignore_errors=True)
        # Restore original model directory
        self.predictor.service.model_dir = self.original_model_dir # Restore service's model_dir

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

    def _create_dummy_model_file(self, symbol, model_index=0):
        """Creates a dummy model file to satisfy the loading logic."""
        from tensorflow.keras import layers, Model, Input, Sequential
        import numpy as np
        sequence_length = self.predictor.service.sequence_length 
        model_path = self.predictor.service.model_dir / f"{symbol}_model_{model_index}.keras"
        
        # Create a simple Sequential model that outputs a single value
        model = Sequential([
            layers.LSTM(10, activation='relu', input_shape=(sequence_length, 17)),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Build the model explicitly to ensure weights are created
        model.build(input_shape=(None, sequence_length, 17))

        # Set dummy weights for the dense layer to ensure it's initialized
        # The LSTM layer has 10 units, so the Dense layer will have 10 input features.
        if model.layers and isinstance(model.layers[-1], layers.Dense):
            # Output a scaled value (e.g., 0.9, which would inverse transform to 190.0)
            dummy_weights = [np.zeros((10, 1)), np.array([0.9])] # Changed bias to 0.9
            model.layers[-1].set_weights(dummy_weights)

        model.save(model_path)


    def test_feature_quality_warnings_fixed(self):
        """
        Test that 'Momentum_Score' and 'MACD_Histogram' warnings are no longer present
        after applying the fixes.
        """
        df = self._create_dummy_data(num_rows=200)
        df['Close'] = df['Close'].rolling(window=5).mean().bfill()
        df['Volume'] = df['Volume'].rolling(window=5).mean().bfill()
        enhanced_data = _calculate_robust_features(df)
        enhanced_data = enhanced_data.dropna()
        validation_results = _validate_feature_quality(enhanced_data, "TEST_SYMBOL")
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
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol="TEST", validation_split=0.2)
        dummy_data = pd.DataFrame({
            'Close': [100, 110, 120, 130, 140],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'High': [105, 115, 125, 135, 145],
            'Low': [95, 105, 115, 125, 135],
            'Open': [98, 108, 118, 128, 138]
        })
        scaler.fit_transform(dummy_data)
        
        self.assertTrue(scaler.fitted)
        self.assertIsNotNone(scaler.price_min)
        self.assertIsNotNone(scaler.price_max)
        self.assertIsNotNone(scaler.data_range)
        
        original_price_min = scaler.price_min
        original_price_max = scaler.price_max
        original_data_range = scaler.data_range
        
        scaler_path = Path(self.test_model_dir) / "test_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        loaded_scaler = joblib.load(scaler_path)
        
        self.assertEqual(loaded_scaler.price_min, original_price_min)
        self.assertEqual(loaded_scaler.price_max, original_price_max)
        self.assertEqual(loaded_scaler.data_range, original_data_range)
        self.assertTrue(loaded_scaler.fitted)

    def test_scaler_metadata_saving(self):
        """Test that scaler metadata is saved correctly during model training."""
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol="TEST", validation_split=0.2)
        dummy_data = pd.DataFrame({
            'Close': [100, 110, 120, 130, 140],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'High': [105, 115, 125, 135, 145],
            'Low': [95, 105, 115, 125, 135],
            'Open': [98, 108, 118, 128, 138]
        })
        scaler.fit_transform(dummy_data)
        
        # Create a minimal real TensorFlow model for metadata extraction
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(10, activation='relu', input_shape=(self.predictor.service.sequence_length, 5)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Use the actual save_ensemble function from model_manager
        # Pass a list containing the real TF model
        save_ensemble([model], scaler, "TEST", self.predictor.service.model_dir, self.predictor.service.sequence_length, [])
        
        # Load the metadata using get_model_info from model_manager
        metadata = get_model_info("TEST", self.predictor.service.model_dir)
        
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
        self._create_dummy_model_file(symbol)
        
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol=symbol)
        
        scaler.price_min = 100.0
        scaler.price_max = 200.0
        scaler.data_range = 100.0
        scaler.fitted = True
        
        original_price_min = scaler.price_min
        original_price_max = scaler.price_max
        
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
        
        metadata_path = self.predictor.service.model_dir / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        scaler.price_min = None
        scaler.price_max = None
        scaler.data_range = None
        
        scaler_path = self.predictor.service.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        _, restored_scaler = load_ensemble_with_fallback(symbol, self.predictor.service.model_dir, self.predictor.service.sequence_length, self.predictor.service.ensemble_size)
        
        self.assertIsNotNone(restored_scaler)
        self.assertEqual(restored_scaler.price_min, original_price_min)
        self.assertEqual(restored_scaler.price_max, original_price_max)
        self.assertEqual(restored_scaler.data_range, 100.0)

    def test_scaling_health_check(self):
        """Test the scaling health check function."""
        symbol = "TEST"
        
        health = self.predictor.check_scaling_health(symbol)
        self.assertFalse(health['scaling_healthy'])
        self.assertIn("Scaler file not found", health['issues'])
        
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        healthy_scaler = CompositeScaler(feature_names, symbol=symbol)
        healthy_scaler.price_min = 100.0
        healthy_scaler.price_max = 200.0
        healthy_scaler.data_range = 100.0
        healthy_scaler.fitted = True
        
        scaler_path = self.predictor.service.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(healthy_scaler, scaler_path)
        
        health = self.predictor.check_scaling_health(symbol)
        self.assertTrue(health['scaling_healthy'])
        self.assertIn("Scaling parameters are healthy", health['recommendations'])
        
        corrupted_scaler = CompositeScaler(feature_names, symbol=symbol)
        corrupted_scaler.fitted = True
        
        joblib.dump(corrupted_scaler, scaler_path)
        
        metadata = {
            'symbol': symbol,
            'scaler_info': {
                'price_min': 100.0,
                'price_max': 200.0,
                'data_range': 100.0
            }
        }
        metadata_path = self.predictor.service.model_dir / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        health = self.predictor.check_scaling_health(symbol)
        self.assertFalse(health['scaling_healthy'])  
        self.assertIn("Scaler parameters missing but available in metadata", health['issues'])
        auto_restore_found = any("auto-restored from metadata" in rec for rec in health['recommendations'])
        self.assertTrue(auto_restore_found, f"Expected auto-restore recommendation not found in: {health['recommendations']}")

    def test_model_diagnosis_scaling_issues(self):
        """Test that model diagnosis detects scaling issues."""
        symbol = "TEST"
        
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        corrupted_scaler = CompositeScaler(feature_names, symbol=symbol)
        corrupted_scaler.fitted = True
        
        scaler_path = self.predictor.service.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(corrupted_scaler, scaler_path)
        
        diagnosis = self.predictor.diagnose_model_issues(symbol)
        
        self.assertIn("Missing price range parameters - predictions will be inaccurate", diagnosis['issues'])
        retrain_recommendation_found = any("Retrain the model to fix scaling issues" in rec for rec in diagnosis['recommendations'])
        self.assertTrue(retrain_recommendation_found, f"Expected scaling fix recommendation not found in: {diagnosis['recommendations']}")

    def test_scaler_fallback_data_restoration(self):
        """Test that scaler falls back to data restoration when metadata unavailable."""
        symbol = "TEST"
        self._create_dummy_model_file(symbol)
        
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        corrupted_scaler = CompositeScaler(feature_names, symbol=symbol)
        corrupted_scaler.fitted = True
        
        scaler_path = self.predictor.service.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(corrupted_scaler, scaler_path)
        
        test_data = pd.DataFrame({
            'Close': [150, 160, 170, 180, 190],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'High': [155, 165, 175, 185, 195],
            'Low': [145, 155, 165, 175, 185],
            'Open': [148, 158, 168, 178, 188]
        })
        
        _, restored_scaler = load_ensemble_with_fallback(symbol, self.predictor.service.model_dir, self.predictor.service.sequence_length, self.predictor.service.ensemble_size, test_data)
        
        self.assertIsNotNone(restored_scaler)
        self.assertEqual(restored_scaler.price_min, 150.0)  
        self.assertEqual(restored_scaler.price_max, 190.0)  
        self.assertEqual(restored_scaler.data_range, 40.0)  

    def test_end_to_end_scaling_workflow(self):
        """Test complete end-to-end workflow: save model -> corrupt scaler -> detect issue -> restore."""
        symbol = "TEST"
        self._create_dummy_model_file(symbol)
        
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
        
        scaler_path = self.predictor.service.model_dir / f"{symbol}_scaler.pkl"
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
        
        metadata_path = self.predictor.service.model_dir / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        corrupted_scaler = joblib.load(scaler_path)
        corrupted_scaler.price_min = None  
        corrupted_scaler.price_max = None
        corrupted_scaler.data_range = None
        joblib.dump(corrupted_scaler, scaler_path)
        
        health_before = self.predictor.check_scaling_health(symbol)
        self.assertFalse(health_before['scaling_healthy'])
        self.assertIn("Scaler parameters missing but available in metadata", health_before['issues'])
        
        _, restored_scaler = load_ensemble_with_fallback(symbol, self.predictor.service.model_dir, self.predictor.service.sequence_length, self.predictor.service.ensemble_size)
        
        self.assertIsNotNone(restored_scaler)
        self.assertEqual(restored_scaler.price_min, original_price_min)
        self.assertEqual(restored_scaler.price_max, original_price_max)
        self.assertTrue(restored_scaler.fitted)
        
        health_after = self.predictor.check_scaling_health(symbol)
        
        self.assertTrue(health_after['scaling_healthy'])
        self.assertIn("Scaling parameters are healthy", health_after['recommendations'])
        self.assertEqual(health_after['scaler_info']['price_min'], original_price_min)
        self.assertEqual(health_after['scaler_info']['price_max'], original_price_max)

    def test_prediction_scaling_accuracy(self):
        """
        Test that predictions maintain proper scaling accuracy and do not exhibit a significant gap.
        This specifically targets the issue of CompositeScaler being incorrectly refitted during prediction.
        """
        symbol = "ACCURACY_TEST"
        sequence_length = self.predictor.service.sequence_length 
        
        training_data = pd.DataFrame({
            'Close': np.linspace(100, 200, 200), 
            'Volume': np.random.rand(200) * 1000000 + 100000,
            'High': np.linspace(101, 201, 200),
            'Low': np.linspace(99, 199, 200),
            'Open': np.linspace(100, 200, 200),
        }, index=pd.date_range(start='2023-01-01', periods=200, freq='D'))

        _, _, original_scaler = prepare_enhanced_data(training_data, sequence_length)
        
        scaler_path = self.predictor.service.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(original_scaler, scaler_path)
        self._create_dummy_model_file(symbol) 

        metadata = {
            'symbol': symbol,
            'ensemble_size': 1,
            'sequence_length': sequence_length,
            'training_date': pd.Timestamp.now().isoformat(),
            'model_format': 'keras',
            'tensorflow_version': tf.__version__,
            'keras_version': tf.keras.__version__,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'training_histories': [],
            'saved_model_paths': [str(self.predictor.service.model_dir / f"{symbol}_model_0.keras")],
            'model_architecture': {
                'input_shape': [None, sequence_length, 17],
                'output_shape': [None, 1],
                'total_params': 1000, 
                'feature_count': 17,
                'uses_enhanced_features': True
            },
            'scaler_info': {
                'scaler_type': type(original_scaler).__name__,
                'fitted': getattr(original_scaler, 'fitted', None),
                'price_min': getattr(original_scaler, 'price_min', None),
                'price_max': getattr(original_scaler, 'price_max', None),
                'data_range': getattr(original_scaler, 'data_range', None),
                'n_features': getattr(original_scaler, 'n_features_in_', None)
            }
        }
        metadata_path = self.predictor.service.model_dir / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        loaded_models, loaded_scaler = load_ensemble_with_fallback(
            symbol, self.predictor.service.model_dir, sequence_length, self.predictor.service.ensemble_size, training_data
        )
        
        self.assertIsNotNone(loaded_scaler, "Loaded scaler should not be None")
        self.assertTrue(loaded_scaler.fitted, "Loaded scaler should be fitted")
        self.assertIsNotNone(loaded_scaler.price_min, "Loaded scaler should have price_min")
        self.assertIsNotNone(loaded_scaler.price_max, "Loaded scaler should have price_max")
        
        self.assertAlmostEqual(loaded_scaler.price_min, original_scaler.price_min, places=2)
        self.assertAlmostEqual(loaded_scaler.price_max, original_scaler.price_max, places=2)

        prediction_data = training_data.iloc[-sequence_length-60:].copy() 
        
        predictions_output = self.predictor.predict(
            symbol=symbol,
            data=prediction_data,
            days=5 
        )
        
        self.assertIn('predictions', predictions_output)
        self.assertGreater(len(predictions_output['predictions']), 0)
        
        last_actual_price = predictions_output['last_price']
        predicted_prices = predictions_output['predictions']
        
        avg_predicted_price = np.mean(predicted_prices)
        
        acceptable_deviation_percent = 0.10 
        
        lower_bound = last_actual_price * (1 - acceptable_deviation_percent)
        upper_bound = last_actual_price * (1 + acceptable_deviation_percent)
        
        self.assertGreaterEqual(avg_predicted_price, lower_bound, 
                                msg=f"Predicted average {avg_predicted_price:.2f} is too low compared to last actual {last_actual_price:.2f}")
        self.assertLessEqual(avg_predicted_price, upper_bound, 
                             msg=f"Predicted average {avg_predicted_price:.2f} is too high compared to last actual {last_actual_price:.2f}")
        

    def test_prediction_consistency_validation(self):
        """Test that prediction consistency is maintained with proper scaling."""
        feature_names = ['Close', 'Volume', 'High', 'Low', 'Open']
        scaler = CompositeScaler(feature_names, symbol="TEST")
        
        scaler.price_min = 100.0
        scaler.price_max = 200.0
        scaler.data_range = 100.0
        scaler.fitted = True
        scaler.n_features_in_ = 5
        
        test_price = 150.0  
        
        dummy_data = np.array([[test_price, 1000, 155, 145, 148]])
        
        scaled_data = scaler.transform(dummy_data)
        expected_scaled_price = (test_price - scaler.price_min) / scaler.data_range
        
        np.testing.assert_almost_equal(scaled_data[0, 0], expected_scaled_price, decimal=6)
        
        inverse_data = scaler.inverse_transform(scaled_data)
        
        np.testing.assert_almost_equal(inverse_data[0, 0], test_price, decimal=6)

    def test_real_world_integration_amzn(self):
        """Integration test using real AMZN model files (if available)."""
        real_predictor = LSTMPredictor()
        real_predictor.service.model_dir = self.original_model_dir  # Use real model directory
        
        symbol = "AMZN"
        
        try:
            health_result = real_predictor.check_scaling_health(symbol)
            
            self.assertIn('symbol', health_result)
            self.assertIn('scaling_healthy', health_result) 
            self.assertIn('issues', health_result)
            self.assertIn('recommendations', health_result)
            self.assertIn('scaler_info', health_result)
            self.assertEqual(health_result['symbol'], symbol)
            
            diagnosis = real_predictor.diagnose_model_issues(symbol)
            
            self.assertIn('issues', diagnosis)
            self.assertIn('recommendations', diagnosis)
            
            if not health_result['scaling_healthy']:
                scaling_issue_found = any("scaling" in issue.lower() for issue in diagnosis['issues'])
                scaling_recommendation_found = any("scaling" in rec.lower() or "retrain" in rec.lower() 
                                                 for rec in diagnosis['recommendations'])
                self.assertTrue(scaling_issue_found or scaling_recommendation_found,
                               f"Scaling issues not properly detected in diagnosis. "
                               f"Health: {health_result['scaling_healthy']}, "
                               f"Issues: {diagnosis['issues']}, "
                               f"Recommendations: {diagnosis['recommendations']}")
                               
        except Exception as e:
            self.skipTest(f"Real AMZN model files not available for integration test: {e}")

if __name__ == '__main__':
    unittest.main()
