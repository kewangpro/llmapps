import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
import logging

from src.config import Config

logger = logging.getLogger(__name__)

class LSTMPredictor:
    """LSTM ensemble for stock price prediction"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Config.MODEL_DIR / "lstm"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = Config.LSTM_SEQUENCE_LENGTH
        self.ensemble_size = Config.LSTM_ENSEMBLE_SIZE
        self.models = []
        self.scalers = []
        
        # Configure TensorFlow for better performance on MacOS
        try:
            tf.config.set_visible_devices([], 'GPU')  # Force CPU usage for stability
        except RuntimeError:
            # Already configured
            pass
        
        try:
            tf.config.threading.set_intra_op_parallelism_threads(0)
            tf.config.threading.set_inter_op_parallelism_threads(0)
        except RuntimeError:
            # Already configured
            pass
        
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create a single LSTM model with explicit functions for better serialization"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Use explicit function objects instead of strings for better serialization
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for LSTM training"""
        # Use multiple features for better predictions
        features = ['Close', 'Volume', 'High', 'Low', 'Open']
        
        # Ensure all required columns exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        dataset = data[features].values
        
        # Handle any NaN values
        if np.isnan(dataset).any():
            logger.warning("NaN values detected in data, filling with forward fill")
            dataset = pd.DataFrame(dataset).ffill().values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict only the close price (first feature)
        
        return np.array(X), np.array(y), scaler
    
    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a proper datetime index"""
        try:
            # If index is already datetime, normalize and return
            if isinstance(data.index, pd.DatetimeIndex):
                return data
            
            # If index is not datetime, try to convert
            if hasattr(data.index, 'dtype') and data.index.dtype == 'object':
                # Try to convert string dates
                data.index = pd.to_datetime(data.index)
                return data
            
            # If index appears to be numeric/positional, create a date range
            # This is a fallback for data without proper date index
            logger.warning("Data index is not datetime, creating synthetic datetime index")
            end_date = pd.Timestamp.now().normalize()
            start_date = end_date - pd.Timedelta(days=len(data)-1)
            
            # Create new DataFrame with datetime index
            new_index = pd.date_range(start=start_date, end=end_date, periods=len(data))
            data_with_datetime = data.copy()
            data_with_datetime.index = new_index
            
            return data_with_datetime
            
        except Exception as e:
            logger.warning(f"Failed to ensure datetime index: {e}")
            # Last resort: create a simple date range
            end_date = pd.Timestamp.now().normalize()
            start_date = end_date - pd.Timedelta(days=len(data)-1)
            new_index = pd.date_range(start=start_date, end=end_date, periods=len(data))
            
            data_copy = data.copy()
            data_copy.index = new_index
            return data_copy
    
    def train_ensemble(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        validation_split: float = 0.2,
        epochs: int = None,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """Train ensemble of LSTM models"""
        
        epochs = epochs or Config.EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE
        
        logger.info(f"Training LSTM ensemble for {symbol}")
        
        # Ensure data has proper datetime index before processing
        data = self._ensure_datetime_index(data)
        
        # Prepare data
        X, y, scaler = self.prepare_data(data)
        
        if len(X) < 50:  # Need minimum data for training
            raise ValueError(f"Insufficient data for training: {len(X)} samples. Need at least 50.")
        
        # Split data
        split_index = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        
        models = []
        training_histories = []
        
        # Train ensemble
        for i in range(self.ensemble_size):
            logger.info(f"Training model {i+1}/{self.ensemble_size}")
            
            model = self.create_lstm_model((X.shape[1], X.shape[2]))
            
            # Early stopping and model checkpointing
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=1e-7
                )
            ]
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            models.append(model)
            training_histories.append(history.history)
        
        # Save models and metadata
        self._save_ensemble(models, scaler, symbol, training_histories)
        
        # Calculate validation metrics
        val_predictions = self._predict_ensemble(models, X_val, scaler)
        actual_prices = self._inverse_transform_predictions(y_val, scaler)
        
        mse = mean_squared_error(actual_prices, val_predictions)
        mae = mean_absolute_error(actual_prices, val_predictions)
        
        # Calculate directional accuracy
        actual_directions = np.diff(actual_prices) > 0
        pred_directions = np.diff(val_predictions) > 0
        directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        
        metrics = {
            'symbol': symbol,
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'directional_accuracy': float(directional_accuracy),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'ensemble_size': self.ensemble_size
        }
        
        logger.info(f"Training completed for {symbol}. RMSE: {metrics['rmse']:.4f}, Directional Accuracy: {directional_accuracy:.1f}%")
        
        return metrics
    
    def predict(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        days: int = None
    ) -> Dict[str, Any]:
        """Generate predictions using trained ensemble with improved error handling"""
        
        days = days or Config.PREDICTION_DAYS
        
        try:
            # Load models with error handling and automatic fallback
            models, scaler = self._load_ensemble_with_fallback(symbol, data)
            if not models:
                # Try to diagnose the issue
                diagnosis = self.diagnose_model_issues(symbol)
                error_msg = f"No trained models found for {symbol}. Issues: {diagnosis['issues']}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as load_error:
            logger.error(f"Failed to load models for {symbol}: {load_error}")
            # Provide helpful error message with recovery suggestions
            raise ValueError(
                f"Could not load models for {symbol}. "
                f"Error: {str(load_error)}. "
                f"Try: 1) Retraining the model, 2) Check diagnose_model_issues() output, "
                f"3) Use force_retrain_if_broken() method."
            )
        
        # Ensure data has proper datetime index before processing
        data = self._ensure_datetime_index(data)
        
        # Prepare recent data for prediction
        X, _, _ = self.prepare_data(data)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Use the last sequence for prediction
        last_sequence = X[-1:, :, :]
        
        # Generate predictions for each day
        predictions = []
        prediction_sequences = []
        current_sequence = last_sequence.copy()
        
        for day in range(days):
            # Get ensemble prediction for current sequence
            day_prediction = self._predict_ensemble(models, current_sequence, scaler)
            predictions.append(day_prediction[0])
            prediction_sequences.append(current_sequence.copy())
            
            # Update sequence for next prediction
            # This is a simplified approach - we use the predicted price to update the sequence
            new_row = current_sequence[0, -1, :].copy()
            
            # Normalize the predicted price change relative to the previous prediction
            if len(predictions) > 1:
                price_change = (day_prediction[0] - predictions[-2]) / predictions[-2]
            else:
                # Use the change from last actual price
                last_actual_price = self._inverse_transform_predictions(
                    np.array([X[-1, -1, 0]]), scaler
                )[0]
                price_change = (day_prediction[0] - last_actual_price) / last_actual_price
            
            # Update the close price feature (index 0) in the new row
            new_row[0] = max(0.01, min(0.99, new_row[0] + price_change * 0.1))  # Conservative update
            
            # Roll the sequence and add the new row
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
        
        # Calculate confidence intervals based on ensemble variance
        ensemble_predictions = []
        for model in models:
            model_preds = []
            seq = last_sequence.copy()
            for day in range(days):
                pred = model.predict(seq, verbose=0)
                pred_price = self._inverse_transform_predictions(pred.flatten(), scaler)[0]
                model_preds.append(pred_price)
                # Update sequence similar to above
                new_row = seq[0, -1, :].copy()
                if len(model_preds) > 1:
                    price_change = (pred_price - model_preds[-2]) / model_preds[-2]
                else:
                    last_actual_price = self._inverse_transform_predictions(
                        np.array([X[-1, -1, 0]]), scaler
                    )[0]
                    price_change = (pred_price - last_actual_price) / last_actual_price
                new_row[0] = max(0.01, min(0.99, new_row[0] + price_change * 0.1))
                seq = np.roll(seq, -1, axis=1)
                seq[0, -1, :] = new_row
            ensemble_predictions.append(model_preds)
        
        # Calculate statistics across ensemble
        ensemble_predictions = np.array(ensemble_predictions)
        pred_std = np.std(ensemble_predictions, axis=0)
        
        # Generate dates for predictions with proper datetime handling
        last_date = data.index[-1]
        
        # Ensure last_date is a proper datetime object
        if isinstance(last_date, (int, float, str)):
            # Handle cases where index is not datetime
            if isinstance(last_date, str):
                try:
                    last_date = pd.to_datetime(last_date)
                except:
                    # If string conversion fails, use current date
                    last_date = pd.Timestamp.now().normalize()
            else:
                # If it's numeric, assume it's days since some epoch - use current date instead
                last_date = pd.Timestamp.now().normalize()
        elif not isinstance(last_date, (pd.Timestamp, datetime)):
            # Convert to pandas Timestamp for consistency
            last_date = pd.to_datetime(last_date)
        
        # Normalize to remove time component if present
        if hasattr(last_date, 'normalize'):
            last_date = last_date.normalize()
        
        # Generate prediction dates starting from the next business day
        # Use proper datetime arithmetic for pandas 2.0+ compatibility
        start_date = last_date + pd.Timedelta(days=1)
        prediction_dates = pd.date_range(
            start=start_date,
            periods=days,
            freq='D'
        )
        
        result = {
            'symbol': symbol,
            'predictions': predictions,
            'dates': prediction_dates.strftime('%Y-%m-%d').tolist(),
            'confidence_upper': [p + 1.96 * s for p, s in zip(predictions, pred_std)],
            'confidence_lower': [p - 1.96 * s for p, s in zip(predictions, pred_std)],
            'last_price': float(data['Close'].iloc[-1]),
            'prediction_period_days': days,
            'prediction_variance': pred_std.tolist()
        }
        
        return result
    
    def _predict_ensemble(self, models: list, X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        """Get ensemble prediction"""
        predictions = []
        
        for model in models:
            pred = model.predict(X, verbose=0)
            pred_rescaled = self._inverse_transform_predictions(pred.flatten(), scaler)
            predictions.append(pred_rescaled)
        
        # Average ensemble predictions
        return np.mean(predictions, axis=0)
    
    def _inverse_transform_predictions(self, predictions: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        """Inverse transform predictions to original scale"""
        # Create dummy array for inverse transform
        dummy = np.zeros((len(predictions), scaler.n_features_in_))
        dummy[:, 0] = predictions
        
        inverse_transformed = scaler.inverse_transform(dummy)
        return inverse_transformed[:, 0]
    
    def _save_ensemble(self, models: list, scaler: MinMaxScaler, symbol: str, histories: list):
        """Save ensemble models and metadata using modern .keras format"""
        try:
            # Migrate any existing .h5 models to .keras format first
            self._migrate_h5_to_keras(symbol)
            
            # Save models using the new .keras format (TensorFlow 2.20.0 compatible)
            saved_models = []
            for i, model in enumerate(models):
                model_path = self.model_dir / f"{symbol}_model_{i}.keras"
                try:
                    # Ensure model is properly compiled before saving
                    self._ensure_model_compiled(model, symbol, i)
                    
                    # Use .keras extension for TensorFlow 2.20.0+ (no save_format parameter needed)
                    model.save(model_path)
                    
                    # Verify the save worked by attempting to load
                    try:
                        test_model = tf.keras.models.load_model(
                            model_path, 
                            custom_objects=self._get_comprehensive_custom_objects(),
                            compile=False
                        )
                        if test_model is not None:
                            logger.debug(f"Successfully saved and verified model {i} for {symbol} in .keras format")
                            saved_models.append(model_path)
                        else:
                            raise ValueError("Model loaded as None")
                    except Exception as verify_error:
                        logger.warning(f"Model {i} saved but failed verification: {verify_error}")
                        # Remove potentially corrupted file
                        if model_path.exists():
                            model_path.unlink()
                        raise verify_error
                        
                except Exception as model_save_error:
                    logger.error(f"Failed to save model {i} for {symbol} in .keras format: {model_save_error}")
                    
                    # Fallback strategy 1: Try SavedModel format (properly handled for TF 2.20.0)
                    savedmodel_path = self.model_dir / f"{symbol}_model_{i}_savedmodel"
                    try:
                        logger.info(f"Attempting SavedModel export fallback for model {i} for {symbol}")
                        # Use export for SavedModel format in TensorFlow 2.20.0+
                        if hasattr(model, 'export'):
                            model.export(savedmodel_path)
                        else:
                            # Fallback for older TensorFlow versions
                            tf.saved_model.save(model, str(savedmodel_path))
                        
                        # Verify SavedModel
                        test_model = tf.keras.models.load_model(
                            savedmodel_path,
                            custom_objects=self._get_comprehensive_custom_objects()
                        )
                        if test_model is not None:
                            logger.warning(f"Saved model {i} for {symbol} in SavedModel format as fallback")
                            saved_models.append(savedmodel_path)
                        else:
                            raise ValueError("SavedModel loaded as None")
                            
                    except Exception as savedmodel_error:
                        logger.error(f"SavedModel fallback failed for model {i}: {savedmodel_error}")
                        
                        # Fallback strategy 2: Try legacy .h5 format (last resort)
                        h5_path = self.model_dir / f"{symbol}_model_{i}.h5"
                        try:
                            logger.warning(f"Attempting legacy .h5 fallback for model {i} for {symbol}")
                            # For TensorFlow 2.20.0, avoid deprecated save_format parameter
                            if h5_path.suffix == '.h5':
                                # Save as HDF5 by using the .h5 extension
                                model.save(h5_path)
                            else:
                                model.save(str(h5_path))
                                
                            # Verify H5 model
                            test_model = tf.keras.models.load_model(
                                h5_path,
                                custom_objects=self._get_comprehensive_custom_objects(),
                                compile=False
                            )
                            if test_model is not None:
                                logger.warning(f"Saved model {i} for {symbol} in .h5 format as final fallback")
                                saved_models.append(h5_path)
                            else:
                                raise ValueError("H5 model loaded as None")
                                
                        except Exception as h5_error:
                            logger.error(f"All save formats failed for model {i}: {h5_error}")
                            raise Exception(f"Could not save model {i} in any format: .keras={model_save_error}, SavedModel={savedmodel_error}, .h5={h5_error}")
            
            # Verify we saved at least some models
            if not saved_models:
                raise Exception("No models were successfully saved")
            
            logger.info(f"Successfully saved {len(saved_models)}/{len(models)} models for {symbol}")
            
            # Save scaler
            scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
            try:
                joblib.dump(scaler, scaler_path)
                logger.debug(f"Saved scaler for {symbol}")
            except Exception as scaler_error:
                logger.error(f"Failed to save scaler for {symbol}: {scaler_error}")
                raise
            
            # Save metadata with enhanced format information
            metadata = {
                'symbol': symbol,
                'ensemble_size': len(models),
                'models_saved': len(saved_models),
                'sequence_length': self.sequence_length,
                'training_date': pd.Timestamp.now().isoformat(),
                'model_format': 'keras',
                'tensorflow_version': tf.__version__,
                'keras_version': tf.keras.__version__,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'training_histories': histories,
                'saved_model_paths': [str(p) for p in saved_models],
                'model_architecture': {
                    'input_shape': list(models[0].input_shape) if models else None,
                    'output_shape': list(models[0].output_shape) if models else None,
                    'total_params': models[0].count_params() if models else None
                }
            }
            
            metadata_path = self.model_dir / f"{symbol}_metadata.json"
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.debug(f"Saved metadata for {symbol}")
            except Exception as metadata_error:
                logger.warning(f"Failed to save metadata for {symbol}: {metadata_error}")
                # Don't fail the entire save operation for metadata issues
                
            logger.info(f"Successfully saved ensemble models for {symbol} ({len(saved_models)} models)")
            
        except Exception as e:
            logger.error(f"Failed to save models for {symbol}: {e}")
            # Clean up any partially created files
            self._cleanup_failed_save(symbol)
            raise
    
    def _load_ensemble(self, symbol: str) -> Tuple[list, Optional[MinMaxScaler]]:
        """Load ensemble models and scaler with proper error handling and format support"""
        models = []
        scaler = None
        
        try:
            # Load scaler
            scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                logger.debug(f"Loaded scaler for {symbol}")
            else:
                logger.warning(f"No scaler found for {symbol}")
                return [], None
            
            # Try to migrate old .h5 models to .keras format if needed
            self._migrate_h5_to_keras(symbol)
            
            # Load models with fallback strategy
            for i in range(self.ensemble_size):
                model = self._load_single_model(symbol, i)
                if model is not None:
                    models.append(model)
                else:
                    logger.warning(f"Model {i} not found for {symbol}")
            
            if len(models) == 0:
                logger.warning(f"No models found for {symbol}")
                return [], None
            
            # Validate loaded models
            self._validate_loaded_models(models, symbol)
            
            logger.info(f"Successfully loaded {len(models)} models for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to load models for {symbol}: {e}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)
            return [], None
        
        return models, scaler
    
    def _load_ensemble_with_fallback(self, symbol: str, data: pd.DataFrame = None) -> Tuple[list, Optional[MinMaxScaler]]:
        """Load ensemble with automatic fallback and recovery mechanisms"""
        try:
            # First attempt: normal loading
            models, scaler = self._load_ensemble(symbol)
            if models and len(models) > 0:
                return models, scaler
                
            logger.warning(f"Normal loading failed for {symbol}, attempting recovery...")
            
            # Second attempt: try migration and reload
            if self._attempt_model_recovery(symbol):
                models, scaler = self._load_ensemble(symbol)
                if models and len(models) > 0:
                    logger.info(f"Successfully recovered {len(models)} models for {symbol}")
                    return models, scaler
            
            # Third attempt: force retrain if data is available
            if data is not None and len(data) > 100:  # Ensure sufficient data
                logger.warning(f"Attempting to retrain models for {symbol} due to loading failures")
                if self.force_retrain_if_broken(symbol, data):
                    models, scaler = self._load_ensemble(symbol)
                    if models and len(models) > 0:
                        logger.info(f"Successfully retrained and loaded {len(models)} models for {symbol}")
                        return models, scaler
            
        except Exception as e:
            logger.error(f"All fallback attempts failed for {symbol}: {e}")
        
        return [], None
    
    def _attempt_model_recovery(self, symbol: str) -> bool:
        """Attempt to recover models through migration and cleanup"""
        try:
            # Try migration first
            self._migrate_h5_to_keras(symbol)
            
            # Clean up any corrupted files
            diagnosis = self.diagnose_model_issues(symbol)
            if 'corruption' in str(diagnosis['issues']).lower():
                logger.info(f"Cleaning up potentially corrupted files for {symbol}")
                self._cleanup_failed_save(symbol)
                
            return True
        except Exception as e:
            logger.warning(f"Model recovery attempt failed for {symbol}: {e}")
            return False
    
    def _get_comprehensive_custom_objects(self) -> Dict[str, Any]:
        """Get comprehensive custom objects mapping for legacy model loading"""
        # Import TensorFlow functional API loss/metric functions
        import tensorflow.keras.losses as losses
        import tensorflow.keras.metrics as metrics
        import tensorflow.keras.optimizers as optimizers
        import tensorflow.keras.activations as activations
        
        # TensorFlow 2.20.0 uses lowercase functional names directly
        try:
            mse_func = losses.mse  # tf.keras.losses.mse
            mae_func = metrics.mae  # tf.keras.metrics.mae
        except AttributeError:
            # Fallback for different TensorFlow versions
            mse_func = losses.mean_squared_error
            mae_func = metrics.mean_absolute_error
        
        return {
            # Core loss functions - both string and class mappings
            'mse': mse_func,
            'mae': mae_func,
            'mean_squared_error': mse_func,
            'mean_absolute_error': mae_func,
            'MeanSquaredError': losses.MeanSquaredError(),
            'MeanAbsoluteError': metrics.MeanAbsoluteError(),
            
            # Additional functional mappings for common strings
            'mse_loss': mse_func,
            'mae_loss': mae_func,
            
            # RMSE mappings
            'rmse': metrics.RootMeanSquaredError(),
            'RootMeanSquaredError': metrics.RootMeanSquaredError(),
            
            # Binary and categorical loss functions (common in many models)
            'binary_crossentropy': losses.binary_crossentropy,
            'categorical_crossentropy': losses.categorical_crossentropy,
            'sparse_categorical_crossentropy': losses.sparse_categorical_crossentropy,
            
            # Optimizer mappings with default parameters
            'adam': optimizers.Adam(learning_rate=0.001),
            'Adam': optimizers.Adam(learning_rate=0.001),
            'sgd': optimizers.SGD(learning_rate=0.01),
            'SGD': optimizers.SGD(learning_rate=0.01),
            'rmsprop': optimizers.RMSprop(learning_rate=0.001),
            'RMSprop': optimizers.RMSprop(learning_rate=0.001),
            
            # Activation functions (sometimes needed)
            'relu': activations.relu,
            'sigmoid': activations.sigmoid,
            'tanh': activations.tanh,
            'linear': activations.linear,
            'softmax': activations.softmax,
            
            # Additional metrics that might be saved as strings
            'accuracy': metrics.Accuracy(),
            'precision': metrics.Precision(),
            'recall': metrics.Recall(),
            'auc': metrics.AUC(),
            
            # TensorFlow 2.x function mappings for legacy models
            'tf.keras.losses.mean_squared_error': mse_func,
            'tf.keras.metrics.mean_absolute_error': mae_func,
            'keras.losses.mean_squared_error': mse_func,
            'keras.metrics.mean_absolute_error': mae_func,
            
            # Additional TensorFlow 2.20.0 mappings
            'tf.keras.losses.mse': mse_func,
            'tf.keras.metrics.mae': mae_func,
            'keras.losses.mse': mse_func,
            'keras.metrics.mae': mae_func,
        }
    
    def _load_single_model(self, symbol: str, model_index: int) -> Optional[tf.keras.Model]:
        """Load a single model with comprehensive fallback strategies"""
        model = None
        load_attempts = []
        
        # Get comprehensive custom objects for loading legacy models
        custom_objects = self._get_comprehensive_custom_objects()
        
        # Strategy 1: Try to load .keras format first (preferred)
        keras_path = self.model_dir / f"{symbol}_model_{model_index}.keras"
        if keras_path.exists():
            try:
                logger.debug(f"Attempting to load .keras model {model_index} for {symbol}")
                model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects, compile=False)
                
                # Manually compile the model to ensure proper configuration
                self._ensure_model_compiled(model, symbol, model_index)
                
                # Validate the model works
                if self._validate_single_model(model, model_index, symbol):
                    logger.debug(f"Successfully loaded model {model_index} for {symbol} from .keras format")
                    return model
                else:
                    logger.warning(f"Model {model_index} loaded but failed validation")
                    model = None
                    
            except Exception as e:
                load_attempts.append(f".keras format: {str(e)}")
                logger.warning(f"Failed to load .keras model {model_index} for {symbol}: {e}")
                model = None
        
        # Strategy 2: Try to load .h5 format (legacy) with custom objects
        h5_path = self.model_dir / f"{symbol}_model_{model_index}.h5"
        if h5_path.exists():
            try:
                logger.debug(f"Attempting to load .h5 model {model_index} for {symbol}")
                model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)
                
                # Manually compile the model to ensure proper configuration
                self._ensure_model_compiled(model, symbol, model_index)
                
                # Validate the model works
                if self._validate_single_model(model, model_index, symbol):
                    logger.debug(f"Successfully loaded model {model_index} for {symbol} from .h5 format")
                    return model
                else:
                    logger.warning(f"Model {model_index} loaded but failed validation")
                    model = None
                    
            except Exception as e:
                load_attempts.append(f".h5 format: {str(e)}")
                logger.warning(f"Failed to load .h5 model {model_index} for {symbol}: {e}")
                model = None
        
        # Strategy 3: Try to load SavedModel format (alternative fallback)
        savedmodel_path = self.model_dir / f"{symbol}_model_{model_index}_savedmodel"
        if savedmodel_path.exists():
            try:
                logger.debug(f"Attempting to load SavedModel {model_index} for {symbol}")
                model = tf.keras.models.load_model(savedmodel_path, custom_objects=custom_objects)
                
                # Manually compile the model to ensure proper configuration
                self._ensure_model_compiled(model, symbol, model_index)
                
                # Validate the model works
                if self._validate_single_model(model, model_index, symbol):
                    logger.debug(f"Successfully loaded model {model_index} for {symbol} from SavedModel format")
                    return model
                else:
                    logger.warning(f"Model {model_index} loaded but failed validation")
                    model = None
                    
            except Exception as e:
                load_attempts.append(f"SavedModel format: {str(e)}")
                logger.warning(f"Failed to load SavedModel {model_index} for {symbol}: {e}")
                model = None
        
        # Strategy 4: Try loading with minimal custom objects (for very old models)
        if h5_path.exists() or keras_path.exists():
            try:
                logger.debug(f"Attempting to load model {model_index} for {symbol} with minimal custom objects")
                
                # Use TensorFlow 2.20.0 functional API with error handling
                minimal_custom_objects = {}
                try:
                    minimal_custom_objects = {
                        'mse': tf.keras.losses.mse,
                        'mae': tf.keras.metrics.mae,
                    }
                except AttributeError:
                    # Fallback for different TensorFlow versions
                    minimal_custom_objects = {
                        'mse': tf.keras.losses.mean_squared_error,
                        'mae': tf.keras.metrics.mean_absolute_error,
                    }
                
                load_path = keras_path if keras_path.exists() else h5_path
                model = tf.keras.models.load_model(load_path, custom_objects=minimal_custom_objects, compile=False)
                
                # Manually compile the model
                self._ensure_model_compiled(model, symbol, model_index)
                
                if self._validate_single_model(model, model_index, symbol):
                    logger.info(f"Successfully loaded model {model_index} for {symbol} with minimal custom objects")
                    return model
                else:
                    model = None
                    
            except Exception as e:
                load_attempts.append(f"minimal custom objects: {str(e)}")
                logger.warning(f"Failed to load model {model_index} for {symbol} with minimal custom objects: {e}")
        
        # If all loading strategies failed, log the attempts
        if load_attempts:
            logger.error(f"All loading strategies failed for model {model_index} for {symbol}:")
            for attempt in load_attempts:
                logger.error(f"  - {attempt}")
        
        return None
    
    def _validate_single_model(self, model: tf.keras.Model, model_index: int, symbol: str) -> bool:
        """Validate that a single loaded model is functional"""
        try:
            # Check basic model properties
            if model is None:
                return False
                
            # Check input/output shapes
            if not hasattr(model, 'input_shape') or not hasattr(model, 'output_shape'):
                logger.warning(f"Model {model_index} for {symbol} missing input/output shape")
                return False
            
            # Verify expected input shape
            expected_input_shape = (None, self.sequence_length, 5)
            if model.input_shape != expected_input_shape:
                logger.warning(f"Model {model_index} for {symbol} has unexpected input shape: {model.input_shape} vs {expected_input_shape}")
                return False
            
            # Test with dummy data
            dummy_input = np.zeros((1, self.sequence_length, 5))
            try:
                prediction = model.predict(dummy_input, verbose=0)
                if prediction.shape != (1, 1):
                    logger.warning(f"Model {model_index} for {symbol} has unexpected output shape: {prediction.shape}")
                    return False
                    
                # Check if prediction is reasonable (not NaN or inf)
                if np.isnan(prediction).any() or np.isinf(prediction).any():
                    logger.warning(f"Model {model_index} for {symbol} produces invalid predictions")
                    return False
                    
            except Exception as pred_error:
                logger.warning(f"Model {model_index} for {symbol} failed prediction test: {pred_error}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation failed for model {model_index} for {symbol}: {e}")
            return False
    
    def _ensure_model_compiled(self, model: tf.keras.Model, symbol: str, model_index: int):
        """Ensure model is properly compiled with correct loss and metrics"""
        try:
            # Always recompile the model to ensure consistency
            logger.debug(f"Ensuring proper compilation for model {model_index} for {symbol}")
            
            # Use the same compilation parameters as in create_lstm_model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()]
            )
            
            # Build the model by running a dummy prediction to initialize all metrics
            try:
                # Create dummy input with correct shape (batch_size=1, sequence_length, features)
                dummy_input = np.zeros((1, self.sequence_length, 5))
                
                # Perform dummy prediction to build metrics and ensure model is functional
                dummy_output = model.predict(dummy_input, verbose=0)
                
                # Verify the output shape is correct
                if dummy_output.shape == (1, 1):
                    logger.debug(f"Model {model_index} for {symbol} is properly compiled and functional")
                else:
                    logger.warning(f"Model {model_index} for {symbol} has unexpected output shape: {dummy_output.shape}")
                
                # Additional validation: ensure the model can handle loss computation
                try:
                    dummy_target = np.array([[100.0]])  # Dummy target price
                    # Use the new TensorFlow 2.20.0 API
                    if hasattr(model, 'compute_loss'):
                        loss_value = model.compute_loss(None, dummy_target, dummy_output)
                    else:
                        # Fallback for older versions
                        loss_value = model.compiled_loss(dummy_target, dummy_output)
                    
                    if not np.isnan(loss_value) and not np.isinf(loss_value):
                        logger.debug(f"Model {model_index} loss computation verified: {loss_value}")
                    else:
                        logger.warning(f"Model {model_index} loss computation issue: {loss_value}")
                except Exception as loss_error:
                    logger.debug(f"Could not validate loss computation for model {model_index}: {loss_error}")
                    
            except Exception as prediction_error:
                logger.warning(f"Could not validate model {model_index} for {symbol} with dummy prediction: {prediction_error}")
                # Try to fix by rebuilding the input layer if needed
                try:
                    # Force rebuild by calling the model with proper input
                    model.build(input_shape=(None, self.sequence_length, 5))
                    logger.debug(f"Rebuilt input layer for model {model_index} for {symbol}")
                except Exception as rebuild_error:
                    logger.debug(f"Could not rebuild model {model_index} for {symbol}: {rebuild_error}")
            
            # Verify metrics are properly initialized
            if hasattr(model, 'compiled_metrics') and model.compiled_metrics:
                try:
                    # Check metrics state and reset if needed
                    if hasattr(model.compiled_metrics, 'reset_state'):
                        model.compiled_metrics.reset_state()
                        logger.debug(f"Reset metrics state for model {model_index} for {symbol}")
                except Exception as metrics_reset_error:
                    logger.debug(f"Could not reset metrics for model {model_index}: {metrics_reset_error}")
                    
        except Exception as e:
            logger.warning(f"Could not ensure compilation for model {model_index} for {symbol}: {e}")
            # As a last resort, try basic compilation without metrics
            try:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.MeanSquaredError()
                )
                logger.debug(f"Applied basic compilation for model {model_index} for {symbol}")
            except Exception as basic_compile_error:
                logger.error(f"Even basic compilation failed for model {model_index} for {symbol}: {basic_compile_error}")
    
    def _migrate_h5_to_keras(self, symbol: str):
        """Migrate existing .h5 models to .keras format with proper error handling"""
        try:
            # Use comprehensive custom objects for migration
            custom_objects = self._get_comprehensive_custom_objects()
            
            migrated_count = 0
            for i in range(self.ensemble_size):
                h5_path = self.model_dir / f"{symbol}_model_{i}.h5"
                keras_path = self.model_dir / f"{symbol}_model_{i}.keras"
                
                # Only migrate if .h5 exists and .keras doesn't
                if h5_path.exists() and not keras_path.exists():
                    try:
                        logger.debug(f"Attempting to migrate model {i} for {symbol} from .h5 to .keras")
                        
                        # Load the .h5 model with custom objects
                        model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects)
                        
                        # Create a fresh model with the same architecture but proper compilation
                        # This avoids JSON serialization issues with complex loss functions
                        input_shape = model.input_shape[1:]  # Remove batch dimension
                        fresh_model = self.create_lstm_model(input_shape)
                        
                        # Copy weights from the loaded model to the fresh model
                        try:
                            fresh_model.set_weights(model.get_weights())
                        except Exception as weight_error:
                            logger.warning(f"Could not copy weights for model {i} of {symbol}: {weight_error}")
                            # If weight copying fails, we'll skip this migration
                            continue
                        
                        # Ensure fresh model is properly compiled
                        self._ensure_model_compiled(fresh_model, symbol, i)
                        
                        # Save the fresh model in .keras format (avoids serialization issues)
                        fresh_model.save(keras_path)
                        
                        # Verify the migration worked by attempting to load
                        try:
                            test_model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
                            if test_model is not None:
                                # Verify the weights were transferred correctly
                                if len(test_model.get_weights()) == len(model.get_weights()):
                                    logger.info(f"Successfully migrated model {i} for {symbol} from .h5 to .keras")
                                    migrated_count += 1
                                    
                                    # Automatically remove the old .h5 file after successful migration
                                    try:
                                        h5_path.unlink()
                                        logger.debug(f"Removed legacy .h5 file for model {i} of {symbol}")
                                    except Exception as cleanup_error:
                                        logger.warning(f"Could not remove legacy .h5 file: {cleanup_error}")
                                else:
                                    logger.warning(f"Weight mismatch during migration for model {i} of {symbol}")
                            else:
                                logger.warning(f"Migration verification failed for model {i} of {symbol}")
                        except Exception as verify_error:
                            logger.warning(f"Could not verify migrated model {i} for {symbol}: {verify_error}")
                            # Clean up the potentially corrupted .keras file
                            if keras_path.exists():
                                keras_path.unlink()
                        
                    except Exception as migration_error:
                        logger.warning(f"Failed to migrate model {i} for {symbol}: {migration_error}")
                        # Clean up any partial .keras file
                        if keras_path.exists():
                            try:
                                keras_path.unlink()
                            except:
                                pass
            
            if migrated_count > 0:
                logger.info(f"Successfully migrated {migrated_count} models for {symbol} to .keras format")
            else:
                logger.debug(f"No models migrated for {symbol} (may already be in .keras format)")
                
        except Exception as e:
            logger.error(f"Error during migration for {symbol}: {e}")
    
    def _validate_loaded_models(self, models: list, symbol: str):
        """Validate that loaded models are functional"""
        try:
            for i, model in enumerate(models):
                if model is None:
                    raise ValueError(f"Model {i} is None")
                
                # Check if model has the expected structure
                if not hasattr(model, 'predict'):
                    raise ValueError(f"Model {i} doesn't have predict method")
                
                # Verify input shape expectations
                expected_input_shape = (None, self.sequence_length, 5)  # 5 features
                actual_input_shape = model.input_shape
                
                if actual_input_shape[1:] != expected_input_shape[1:]:
                    logger.warning(
                        f"Model {i} for {symbol} has unexpected input shape. "
                        f"Expected: {expected_input_shape}, Got: {actual_input_shape}"
                    )
                
                logger.debug(f"Model {i} for {symbol} validation passed")
                
        except Exception as e:
            logger.error(f"Model validation failed for {symbol}: {e}")
            raise
    
    def _cleanup_failed_save(self, symbol: str):
        """Clean up partially created files after a failed save operation"""
        try:
            patterns_to_clean = [
                f"{symbol}_model_*.keras",
                f"{symbol}_model_*_savedmodel",
                f"{symbol}_scaler.pkl",
                f"{symbol}_metadata.json"
            ]
            
            for pattern in patterns_to_clean:
                for file_path in self.model_dir.glob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                        logger.debug(f"Cleaned up {file_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up {file_path}: {cleanup_error}")
                        
        except Exception as e:
            logger.error(f"Error during cleanup for {symbol}: {e}")
    
    def is_model_trained(self, symbol: str) -> bool:
        """Check if a trained model exists for the symbol"""
        scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
        
        # Check for models in any supported format
        model_exists = False
        for i in range(self.ensemble_size):
            keras_path = self.model_dir / f"{symbol}_model_{i}.keras"
            h5_path = self.model_dir / f"{symbol}_model_{i}.h5"
            savedmodel_path = self.model_dir / f"{symbol}_model_{i}_savedmodel"
            
            if keras_path.exists() or h5_path.exists() or savedmodel_path.exists():
                model_exists = True
                break
        
        return scaler_path.exists() and model_exists
    
    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about trained model with enhanced details"""
        metadata_path = self.model_dir / f"{symbol}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Add enhanced runtime information
                file_status = self._check_model_files_exist(symbol)
                metadata['model_files_exist'] = file_status
                metadata['current_tensorflow_version'] = tf.__version__
                metadata['current_keras_version'] = tf.keras.__version__
                metadata['compatible'] = self._check_version_compatibility(metadata.get('tensorflow_version'))
                
                # Determine actual format being used
                keras_count = sum(file_status.get('keras_models', []))
                h5_count = sum(file_status.get('h5_models', []))
                savedmodel_count = sum(file_status.get('savedmodel_models', []))
                
                if keras_count > 0:
                    metadata['active_format'] = 'keras'
                elif h5_count > 0:
                    metadata['active_format'] = 'h5_legacy'
                elif savedmodel_count > 0:
                    metadata['active_format'] = 'savedmodel'
                else:
                    metadata['active_format'] = 'none'
                
                metadata['format_counts'] = {
                    'keras': keras_count,
                    'h5': h5_count,
                    'savedmodel': savedmodel_count
                }
                
                # Add migration recommendations
                if h5_count > 0 and keras_count == 0:
                    metadata['migration_recommended'] = True
                    metadata['migration_reason'] = 'Legacy .h5 format detected, migration to .keras recommended'
                else:
                    metadata['migration_recommended'] = False
                
                return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata for {symbol}: {e}")
        return None
    
    def _check_model_files_exist(self, symbol: str) -> Dict[str, bool]:
        """Check which model files exist for a symbol"""
        file_status = {
            'scaler': (self.model_dir / f"{symbol}_scaler.pkl").exists(),
            'metadata': (self.model_dir / f"{symbol}_metadata.json").exists(),
            'keras_models': [],
            'h5_models': [],
            'savedmodel_models': []
        }
        
        for i in range(self.ensemble_size):
            keras_path = self.model_dir / f"{symbol}_model_{i}.keras"
            h5_path = self.model_dir / f"{symbol}_model_{i}.h5"
            savedmodel_path = self.model_dir / f"{symbol}_model_{i}_savedmodel"
            
            file_status['keras_models'].append(keras_path.exists())
            file_status['h5_models'].append(h5_path.exists())
            file_status['savedmodel_models'].append(savedmodel_path.exists())
        
        return file_status
    
    def _check_version_compatibility(self, saved_tf_version: str) -> bool:
        """Check if the saved model is compatible with current TensorFlow version"""
        if not saved_tf_version:
            return True  # Assume compatible if version not recorded
        
        try:
            current_version = tf.__version__.split('.')
            saved_version = saved_tf_version.split('.')
            
            # Check major and minor version compatibility
            current_major_minor = f"{current_version[0]}.{current_version[1]}"
            saved_major_minor = f"{saved_version[0]}.{saved_version[1]}"
            
            return current_major_minor == saved_major_minor
        except Exception:
            return True  # Assume compatible if version parsing fails
    
    def cleanup_old_models(self, symbol: str, keep_keras: bool = True):
        """Clean up old model files, optionally keeping .keras versions"""
        try:
            removed_count = 0
            
            # Remove .h5 files if .keras versions exist
            if keep_keras:
                for i in range(self.ensemble_size):
                    h5_path = self.model_dir / f"{symbol}_model_{i}.h5"
                    keras_path = self.model_dir / f"{symbol}_model_{i}.keras"
                    
                    if h5_path.exists() and keras_path.exists():
                        h5_path.unlink()
                        removed_count += 1
                        logger.info(f"Removed legacy .h5 model {i} for {symbol}")
            
            logger.info(f"Cleaned up {removed_count} old model files for {symbol}")
            
        except Exception as e:
            logger.error(f"Error during cleanup for {symbol}: {e}")
    
    def diagnose_model_issues(self, symbol: str) -> Dict[str, Any]:
        """Diagnose issues with model loading for troubleshooting"""
        diagnosis = {
            'symbol': symbol,
            'issues': [],
            'recommendations': [],
            'file_status': self._check_model_files_exist(symbol),
            'tensorflow_version': tf.__version__
        }
        
        try:
            # Check if scaler exists
            scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
            if not scaler_path.exists():
                diagnosis['issues'].append("Scaler file missing")
                diagnosis['recommendations'].append("Retrain the model to generate scaler")
            
            # Check model files
            models_found = 0
            for i in range(self.ensemble_size):
                keras_path = self.model_dir / f"{symbol}_model_{i}.keras"
                h5_path = self.model_dir / f"{symbol}_model_{i}.h5"
                savedmodel_path = self.model_dir / f"{symbol}_model_{i}_savedmodel"
                
                if not (keras_path.exists() or h5_path.exists() or savedmodel_path.exists()):
                    diagnosis['issues'].append(f"Model {i} missing in all formats")
                else:
                    models_found += 1
                    
                    # Try loading each model to check for issues
                    model = self._load_single_model(symbol, i)
                    if model is None:
                        diagnosis['issues'].append(f"Model {i} exists but cannot be loaded")
                        diagnosis['recommendations'].append(f"Check model {i} corruption, consider retraining")
            
            if models_found == 0:
                diagnosis['issues'].append("No models found")
                diagnosis['recommendations'].append("Train the model first")
            elif models_found < self.ensemble_size:
                diagnosis['issues'].append(f"Only {models_found}/{self.ensemble_size} models found")
                diagnosis['recommendations'].append("Retrain to generate complete ensemble")
            
            # Check metadata
            metadata = self.get_model_info(symbol)
            if metadata:
                if not metadata.get('compatible', True):
                    diagnosis['issues'].append("TensorFlow version mismatch")
                    diagnosis['recommendations'].append("Consider retraining with current TensorFlow version")
            else:
                diagnosis['issues'].append("Metadata file missing")
                diagnosis['recommendations'].append("Metadata will be regenerated on next training")
            
        except Exception as e:
            diagnosis['issues'].append(f"Error during diagnosis: {str(e)}")
            diagnosis['recommendations'].append("Check logs for detailed error information")
        
        return diagnosis
    
    def force_retrain_if_broken(self, symbol: str, data: pd.DataFrame) -> bool:
        """Attempt to retrain models if they cannot be loaded"""
        try:
            # First, try to load existing models
            models, scaler = self._load_ensemble(symbol)
            
            if models and len(models) == self.ensemble_size:
                logger.info(f"Models for {symbol} loaded successfully, no retraining needed")
                return False
            
            # If loading failed, remove corrupted files and retrain
            logger.warning(f"Models for {symbol} are corrupted or missing, initiating retraining")
            
            # Clean up any existing files
            self._cleanup_failed_save(symbol)
            
            # Retrain the ensemble
            metrics = self.train_ensemble(data, symbol)
            
            logger.info(f"Successfully retrained models for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retrain models for {symbol}: {e}")
            return False