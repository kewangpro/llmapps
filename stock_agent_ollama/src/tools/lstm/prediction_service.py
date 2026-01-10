import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.tools.lstm.model_architecture import create_lstm_model
from src.tools.lstm.data_pipeline import _prepare_basic_data, prepare_enhanced_data, prepare_enhanced_data_robust
from src.tools.lstm.model_manager import save_ensemble, load_ensemble, load_ensemble_with_fallback, determine_feature_compatibility, get_model_info
from src.tools.lstm.validation_utils import validate_improvements, check_scaling_health, diagnose_model_issues
from src.tools.lstm.custom_scalers import CompositeScaler
from src.tools.lstm.prediction_utils import _predict_single_model, _inverse_transform_predictions, _update_sequence_for_next_prediction, _generate_multi_step_predictions, _generate_ensemble_predictions # Removed _predict_ensemble

logger = logging.getLogger(__name__)

def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
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


class LSTMProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for reporting LSTM training progress to UI"""

    def __init__(self, progress_callback: Optional[Callable], model_index: int, total_models: int, symbol: str):
        super().__init__()
        self.progress_callback = progress_callback
        self.model_index = model_index
        self.total_models = total_models
        self.symbol = symbol
        self.start_time = None

    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        import time
        self.start_time = time.time()

        if self.progress_callback:
            self.progress_callback({
                'type': 'lstm_training_start',
                'symbol': self.symbol,
                'model': self.model_index + 1,
                'total_models': self.total_models,
                'status': f'Training model {self.model_index + 1}/{self.total_models}'
            })

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        import time
        logs = logs or {}

        if self.progress_callback:
            elapsed_time = time.time() - self.start_time if self.start_time else 0

            self.progress_callback({
                'type': 'lstm_training_progress',
                'symbol': self.symbol,
                'model': self.model_index + 1,
                'total_models': self.total_models,
                'epoch': epoch + 1,
                'loss': float(logs.get('loss', 0)),
                'val_loss': float(logs.get('val_loss', 0)),
                'mae': float(logs.get('mean_absolute_error', 0)),
                'val_mae': float(logs.get('val_mean_absolute_error', 0)),
                'elapsed_time': elapsed_time,
                'status': f'Model {self.model_index + 1}/{self.total_models} - Epoch {epoch + 1}'
            })

    def on_train_end(self, logs=None):
        """Called at the end of training"""
        import time
        logs = logs or {}

        if self.progress_callback:
            elapsed_time = time.time() - self.start_time if self.start_time else 0

            self.progress_callback({
                'type': 'lstm_training_complete',
                'symbol': self.symbol,
                'model': self.model_index + 1,
                'total_models': self.total_models,
                'final_loss': float(logs.get('loss', 0)),
                'final_val_loss': float(logs.get('val_loss', 0)),
                'elapsed_time': elapsed_time,
                'status': f'Completed model {self.model_index + 1}/{self.total_models}'
            })


class LSTMPredictionService:
    """Core service for training, predicting, and managing LSTM models"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Config.MODEL_DIR / "lstm"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = Config.LSTM_SEQUENCE_LENGTH
        self.ensemble_size = Config.LSTM_ENSEMBLE_SIZE

    def _predict_ensemble(self, models: list, X, scaler) -> np.ndarray:
        """Get ensemble prediction using direct model calls without tf.function"""
        predictions = []
        
        # Ensure X is a proper tensor with consistent dtype
        if isinstance(X, np.ndarray):
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        elif isinstance(X, tf.Tensor):
            X_tensor = tf.cast(X, dtype=tf.float32)
        else:
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        for model in models:
            # Use direct model call for better performance without retracing issues
            pred = _predict_single_model(model, X_tensor)
            pred_rescaled = _inverse_transform_predictions(pred.numpy().flatten(), scaler)
            predictions.append(pred_rescaled)
        
        # Average ensemble predictions
        return np.mean(predictions, axis=0)

    def train_ensemble(
        self,
        data: pd.DataFrame,
        symbol: str,
        validation_split: float = 0.2,
        epochs: int = None,
        batch_size: int = None,
        progress_callback: Optional[Callable] = None,
        horizon: int = 1
    ) -> Dict[str, Any]:
        """Train ensemble of LSTM models

        Args:
            data: Stock price DataFrame
            symbol: Stock symbol
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size for training
            progress_callback: Optional callback for progress updates
            horizon: Prediction horizon (1 for next day, 30 for next month, etc.)
        """

        epochs = epochs or Config.EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE
        
        logger.info(f"Training LSTM ensemble for {symbol} (Horizon: {horizon})")
        
        # Ensure data has proper datetime index before processing
        data = _ensure_datetime_index(data)
        
        # Prepare data with enhanced features if possible, fallback to basic features
        try:
            # Try enhanced features first with symbol parameter
            if hasattr(prepare_enhanced_data_robust, '__call__') and symbol in ['MSFT', 'AMZN', 'GOOGL', 'META']:
                logger.info(f"Using robust enhanced features for {symbol} (known outlier issues)")
                X, y, scaler = prepare_enhanced_data_robust(data, self.sequence_length, symbol, validation_split, horizon=horizon)
            else:
                X, y, scaler = prepare_enhanced_data(data, self.sequence_length, validation_split, horizon=horizon)
            logger.info(f"Using enhanced features ({X.shape[2]} features) for training {symbol}")
        except Exception as e:
            logger.warning(f"Enhanced features failed for {symbol}: {e}. Falling back to basic features.")
            X, y, scaler = _prepare_basic_data(data, self.sequence_length, validation_split, horizon=horizon) # Updated call
            logger.info(f"Using basic features ({X.shape[2]} features) for training {symbol}")
        
        if len(X) < 50:  # Need minimum data for training
            raise ValueError(f"Insufficient data for training: {len(X)} samples. Need at least 50.")
        
        # Data is already split in prepare_data methods, so we need to recreate the split
        # IMPLEMENTATION: Walk-Forward Validation (Expanding Window)
        # Instead of a static split, we use an expanding window for each model in the ensemble
        
        total_samples = len(X)
        
        # Calculate validation window size per model
        # We want to cover approx 'validation_split' of the data across the ensemble if possible,
        # or distinct chunks. Let's make each validation fold approx 10-15% of data.
        # But for 'validation_split' parameter consistency, let's use that as the *last* fold size.
        fold_size = int(total_samples * validation_split)
        
        # Ensure minimum fold size
        if fold_size < 20:
            fold_size = 20
        
        models = []
        training_histories = []
        val_predictions_list = []
        val_targets_list = []
        
        # Train ensemble with Walk-Forward Validation
        for i in range(self.ensemble_size):
            # Calculate indices for Expanding Window
            # Model 0: Train [0...T-3V], Val [T-3V...T-2V]
            # Model 1: Train [0...T-2V], Val [T-2V...T-V]
            # Model 2: Train [0...T-V],  Val [T-V...T]
            
            # Offset from the end of the dataset
            # For the last model (i=ensemble_size-1), multiplier is 0 -> split at len(X)-fold_size
            fold_offset = (self.ensemble_size - 1 - i) * fold_size
            
            # The split point (end of training data)
            current_split_index = total_samples - fold_size - fold_offset
            
            # Safety check to ensure we have enough training data
            if current_split_index < 50:
                logger.warning(f"Walk-forward split index {current_split_index} too small. clamping to 50.")
                current_split_index = 50
                # Adjust fold offset if we clamped
                
            X_train = X[:current_split_index]
            y_train = y[:current_split_index]
            
            # Validation set is the next 'fold_size' samples
            # For the last model, this goes to the end of the data.
            val_end = current_split_index + fold_size
            X_val = X[current_split_index:val_end]
            y_val = y[current_split_index:val_end]
            
            logger.info(f"Training model {i+1}/{self.ensemble_size} with Walk-Forward Split: "
                       f"Train [{0}-{current_split_index}], Val [{current_split_index}-{val_end}]")

            model = create_lstm_model((X.shape[1], X.shape[2]))

            # IMPROVED: Enhanced callbacks for better training stability
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=8,  # Reduced from 15 to stop overfitting earlier
                    restore_best_weights=True,
                    min_delta=1e-4,  # Minimum improvement threshold
                    verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,  # Less aggressive reduction
                    patience=8,  # More patience before reducing LR
                    min_lr=1e-7,
                    verbose=0
                ),
                # Add learning rate warmup for first few epochs
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: 0.001 * min(1.0, (epoch + 1) / 5.0),
                    verbose=0
                )
            ]

            # Add progress callback if provided
            if progress_callback:
                callbacks.append(LSTMProgressCallback(progress_callback, i, self.ensemble_size, symbol))
            
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
            
            # Collect validation results for this fold
            fold_preds = self._predict_ensemble([model], X_val, scaler)
            fold_actuals = _inverse_transform_predictions(y_val, scaler)
            val_predictions_list.extend(fold_preds)
            val_targets_list.extend(fold_actuals)
        
        # Save models and metadata
        save_ensemble(models, scaler, symbol, self.model_dir, self.sequence_length, training_histories, horizon)
        
        # Calculate aggregated validation metrics across all walk-forward folds
        all_val_preds = np.array(val_predictions_list)
        all_val_actuals = np.array(val_targets_list)
        
        mse = np.mean((all_val_actuals - all_val_preds)**2)
        mae = np.mean(np.abs(all_val_actuals - all_val_preds))
        
        # Calculate directional accuracy
        actual_directions = np.diff(all_val_actuals) > 0
        pred_directions = np.diff(all_val_preds) > 0
        
        # Handle case where diff might be shorter if folds are disjoint? 
        # Actually diff reduces length by 1. We should calculate per-fold and average, 
        # or concat and accept one missing point at boundaries.
        # Let's calculate simple accuracy on the concatenated array (ignoring boundary effects for simplicity)
        if len(actual_directions) > 0:
            directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        else:
            directional_accuracy = 0.0
        
        metrics = {
            'symbol': symbol,
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'directional_accuracy': float(directional_accuracy),
            'training_samples': len(X) - fold_size, # Approx last training set size
            'validation_samples': len(all_val_actuals),
            'ensemble_size': self.ensemble_size
        }
        
        logger.info(f"Walk-Forward Training completed for {symbol} (Horizon: {horizon}). "
                   f"Aggregated RMSE: {metrics['rmse']:.4f}, Accuracy: {directional_accuracy:.1f}%")
        
        return metrics

    def predict(
        self,
        symbol: str,
        data: pd.DataFrame,
        days: int = None,
        ensemble_size: int = None,
        prediction_callback: Any = None,
        sentiment_score: float = 0.0,
        sentiment_reasoning: str = None
    ) -> Dict[str, Any]:
        """Generate predictions using trained ensemble with improved error handling

        Args:
            symbol: Stock symbol
            data: Historical price data
            days: Number of days to predict
            ensemble_size: Number of models in ensemble
            prediction_callback: Optional callback for prediction progress updates
            sentiment_score: News sentiment score (-1.0 to 1.0)
            sentiment_reasoning: Explanation for the sentiment score
        """

        days = days or Config.PREDICTION_DAYS
        ensemble_size = ensemble_size or self.ensemble_size

        # Progress callback helper
        def report_progress(step: str, progress: int):
            if prediction_callback:
                prediction_callback({
                    'type': 'prediction_progress',
                    'symbol': symbol,
                    'step': step,
                    'progress': progress
                })

        try:
            report_progress("Loading models...", 10)
            # Load standard (h=1) models with error handling and automatic fallback
            models, scaler = load_ensemble_with_fallback(symbol, self.model_dir, self.sequence_length, ensemble_size, data)
            if not models:
                # Try to diagnose the issue
                diagnosis = diagnose_model_issues(symbol, self.model_dir)
                error_msg = f"No trained models found for {symbol}. Issues: {diagnosis['issues']}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            # Do NOT update scaler's price_min and price_max from latest data.
            if scaler is not None:
                logger.info(f"Using training scaler: {scaler.price_min} - {scaler.price_max}")
        except Exception as load_error:
            logger.error(f"Failed to load models for {symbol}: {load_error}")
            raise ValueError(
                f"Could not load models for {symbol}. "
                f"Error: {str(load_error)}. "
                f"Try: 1) Retraining the model, 2) Check diagnose_model_issues() output, "
                f"3) Use force_retrain_if_broken() method."
            )
        
        # Ensure data has proper datetime index before processing
        data = _ensure_datetime_index(data)

        report_progress("Preparing data...", 30)
        # Determine feature set for standard models
        compatibility_info = determine_feature_compatibility(symbol, self.model_dir)
        use_enhanced_features = compatibility_info['uses_enhanced_features']

        logger.info(f"Model for {symbol}: Enhanced features={use_enhanced_features}, "
                   f"Feature count={compatibility_info['feature_count']}")

        # Prepare data for standard models
        try:
            if use_enhanced_features:
                X, _, scaler_check = prepare_enhanced_data(data, self.sequence_length, pre_fitted_scaler=scaler)
            else:
                X, _, scaler_check = _prepare_basic_data(data, self.sequence_length, pre_fitted_scaler=scaler)
                
            if X.shape[2] != compatibility_info['feature_count']:
                logger.warning(f"Feature count mismatch: expected {compatibility_info['feature_count']}, got {X.shape[2]}. Falling back.")
                raise ValueError(f"Feature count mismatch")
                
        except Exception as e:
            logger.warning(f"Feature preparation failed, falling back to basic features: {e}")
            X, _, scaler_check = _prepare_basic_data(data, self.sequence_length)
            
        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")

        report_progress("Generating predictions...", 50)
        # Generate recursive predictions (Standard approach)
        predictions = _generate_multi_step_predictions(models, scaler, X, days, self._predict_ensemble)

        # ---------------------------------------------------------
        # Direct Horizon Blending
        # ---------------------------------------------------------
        if days > 1:
            try:
                # Check if a specific horizon model exists
                direct_metadata = get_model_info(symbol, self.model_dir, horizon=days)
                if direct_metadata:
                    logger.info(f"Found direct horizon model for {symbol} (h={days})")
                    report_progress(f"Refining with h={days} model...", 60)
                    
                    # Load direct models (no fallback to training, just load)
                    direct_models, direct_scaler = load_ensemble(symbol, self.model_dir, self.sequence_length, ensemble_size, data, horizon=days)
                    
                    if direct_models:
                        # Prepare data using direct scaler (use horizon=1 to get latest sequence)
                        # We assume direct models use enhanced features if available
                        if use_enhanced_features: # Assume same architecture preference
                             X_direct, _, _ = prepare_enhanced_data(data, self.sequence_length, pre_fitted_scaler=direct_scaler, horizon=1)
                        else:
                             X_direct, _, _ = _prepare_basic_data(data, self.sequence_length, pre_fitted_scaler=direct_scaler, horizon=1)
                        
                        # Predict single step (the endpoint)
                        last_seq_direct = tf.convert_to_tensor(X_direct[-1:, :, :], dtype=tf.float32)
                        direct_endpoint_scaled = self._predict_ensemble(direct_models, last_seq_direct, direct_scaler)
                        direct_endpoint = direct_endpoint_scaled[0]
                        
                        logger.info(f"Recursive endpoint: {predictions[-1]:.2f}, Direct endpoint: {direct_endpoint:.2f}")
                        
                        # Blend predictions
                        # Adjust the recursive trajectory to end at the direct endpoint
                        recursive_end = predictions[-1]
                        correction_total = direct_endpoint - recursive_end
                        
                        # Apply linear correction over time (confidence in direct model increases as we approach horizon)
                        for i in range(len(predictions)):
                            # i=0 is day 1. i=days-1 is day days.
                            weight = (i + 1) / days
                            predictions[i] += correction_total * weight
                            
            except Exception as direct_error:
                logger.warning(f"Failed to use direct horizon model: {direct_error}")
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Sentiment Adjustment
        # ---------------------------------------------------------
        if sentiment_score != 0.0:
            report_progress("Applying sentiment adjustment...", 65)
            logger.info(f"Applying sentiment adjustment: score={sentiment_score}")
            
            # Max impact at horizon (e.g., 5% for max sentiment)
            MAX_SENTIMENT_IMPACT = 0.05
            
            # Linear scaling of impact over time
            for i in range(len(predictions)):
                # Impact grows over time as sentiment reflects trend
                time_weight = (i + 1) / days
                adjustment_factor = 1.0 + (sentiment_score * MAX_SENTIMENT_IMPACT * time_weight)
                predictions[i] *= adjustment_factor
        # ---------------------------------------------------------

        report_progress("Calculating confidence intervals...", 70)
        # Calculate confidence intervals with improved ensemble variance
        ensemble_predictions = _generate_ensemble_predictions(models, scaler, X, days, self._predict_ensemble)
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
            freq='B'
        )
        
        report_progress("Finalizing results...", 90)
        result = {
            'type': 'prediction',
            'symbol': symbol,
            'predictions': predictions,
            'dates': prediction_dates.strftime('%Y-%m-%d').tolist(),
            'confidence_upper': [p + 1.96 * s for p, s in zip(predictions, pred_std)],
            'confidence_lower': [p - 1.96 * s for p, s in zip(predictions, pred_std)],
            'last_price': float(data['Close'].iloc[-1]),
            'prediction_period_days': days,
            'prediction_variance': pred_std.tolist(),
            'sentiment_analysis': {
                'score': sentiment_score,
                'reasoning': sentiment_reasoning
            } if sentiment_score != 0.0 else None
        }

        report_progress("Complete!", 100)
        return result

    def validate_model(self, data: pd.DataFrame, symbol: str = "TEST") -> Dict[str, Any]:
        """Comprehensive validation of all LSTM predictor improvements"""
        return validate_improvements(data, self.sequence_length, symbol)

    def check_scaling_health(self, symbol: str) -> Dict[str, Any]:
        """Check the health of scaling parameters for a trained model"""
        return check_scaling_health(symbol, self.model_dir)

    def diagnose_model_issues(self, symbol: str) -> Dict[str, Any]:
        """Diagnose potential issues with trained models"""
        return diagnose_model_issues(symbol, self.model_dir)

    def get_model_info(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a trained model"""
        return get_model_info(symbol, self.model_dir)

    def force_retrain_if_broken(self, symbol: str, data: pd.DataFrame) -> bool:
        """Force retrain if model is broken"""
        return load_ensemble_with_fallback(symbol, self.model_dir, self.sequence_length, self.ensemble_size, data)[0] is None
