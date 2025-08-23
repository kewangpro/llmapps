import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from typing import Dict, Any, List, Tuple
import time

logger = logging.getLogger(__name__)

def _predict_single_model(model, X):
    """Direct model prediction without tf.function wrapper"""
    # Ensure consistent input tensor format
    if isinstance(X, np.ndarray):
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    else:
        X_tensor = tf.cast(X, dtype=tf.float32)
    
    # Use direct model call for better performance without retracing
    return model(X_tensor, training=False)

def _inverse_transform_predictions(predictions: np.ndarray, scaler) -> np.ndarray:
    """Inverse transform predictions to original scale"""
    # Create dummy array for inverse transform
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 0] = predictions
    
    inverse_transformed = scaler.inverse_transform(dummy)
    return inverse_transformed[:, 0]

def _update_sequence_for_next_prediction(current_sequence: tf.Tensor, predicted_price: float, 
                                           scaler, original_X: np.ndarray, 
                                           all_predictions: list) -> tf.Tensor:
    """Update sequence for next prediction with proper scaling consistency"""
    seq_np = current_sequence.numpy()
    new_row = seq_np[0, -1, :].copy()
    

    # CRITICAL FIX: Convert predicted price back to scaled space for sequence consistency
    # Use CompositeScaler's min/max for consistent re-scaling
    # Ensure scaler has been fitted and has price_min/price_max
    if not (hasattr(scaler, 'price_min') and scaler.price_min is not None and
            hasattr(scaler, 'data_range') and scaler.data_range is not None and
            scaler.data_range > 0):
        raise ValueError("CompositeScaler price_min or data_range not available for scaling predicted price. This indicates a problem with scaler loading/restoration.")

    # Prevent predicted price from dropping below 98% of last actual price
    if len(all_predictions) > 0:
        last_actual_price = _inverse_transform_predictions(
            np.array([original_X[-1, -1, 0]]), scaler
        )[0]
        min_pred_price = last_actual_price * 0.98
        predicted_price = max(predicted_price, min_pred_price)

    scaled_pred_price = (predicted_price - scaler.price_min) / scaler.data_range
    new_row[0] = scaled_pred_price

    # For enhanced features, estimate other technical indicators
    if new_row.shape[0] > 5:  # Enhanced features
        price_change_ratio = 1.0
        if len(all_predictions) > 0:
            epsilon = 1e-6
            price_change_ratio = predicted_price / (last_actual_price + epsilon)

        # Update volume (index 1) - less aggressive adjustment
        if new_row.shape[0] > 1:
            volume_factor = 1.0 + (price_change_ratio - 1.0) * 0.1 # Adjust volume by 10% of price change
            new_row[1] = np.clip(seq_np[0, -1, 1] * volume_factor, 0.01, 0.99)

        # Update high/low (indices 2,3) based on predicted close and previous day's ratios
        if new_row.shape[0] > 3:
            prev_scaled_close = seq_np[0, -1, 0]
            prev_scaled_high = seq_np[0, -1, 2]
            prev_scaled_low = seq_np[0, -1, 3]

            high_ratio = prev_scaled_high / prev_scaled_close if prev_scaled_close > 0 else 1.01
            low_ratio = prev_scaled_low / prev_scaled_close if prev_scaled_close > 0 else 0.99

            new_row[2] = np.clip(new_row[0] * high_ratio, new_row[0], 0.99)  # High based on ratio
            new_row[3] = np.clip(new_row[0] * low_ratio, 0.01, new_row[0])  # Low based on ratio

        # Keep technical indicators stable (even slower decay and less influence from predicted close)
        for i in range(5, new_row.shape[0]):
            momentum = 0.995  # Keep 99.5% of previous value (much slower decay)
            new_row[i] = seq_np[0, -1, i] * momentum + new_row[0] * (1 - momentum) * 0.05 # Only 5% influence
            new_row[i] = np.clip(new_row[i], 0.01, 0.99)

    # Roll the sequence and add the new row
    seq_np = np.roll(seq_np, -1, axis=1)
    seq_np[0, -1, :] = new_row

    return tf.convert_to_tensor(seq_np, dtype=tf.float32)

def _generate_multi_step_predictions(models: list, scaler, X: np.ndarray, days: int, _predict_ensemble_func) -> list:
    """Generate multi-step predictions with consistent scaling and improved sequence updating"""
    predictions = []
    last_sequence = X[-1:, :, :].copy()
    current_sequence = tf.convert_to_tensor(last_sequence, dtype=tf.float32)
    
    logger.info(f"Starting multi-step prediction for {days} days with {len(models)} models.")
    start_time = time.time() # Add time import
    
    for day in range(days):
        step_start_time = time.time()
        # Get ensemble prediction for current sequence
        day_prediction = _predict_ensemble_func(models, current_sequence, scaler) # Call through passed function
        predictions.append(day_prediction[0])
        
        logger.debug(f"  Day {day+1} prediction: {day_prediction[0]:.2f} (took {time.time() - step_start_time:.4f}s)")
        
        # CRITICAL FIX: Properly update sequence with consistent scaling
        if day < days - 1:  # Don't update sequence for the last prediction
            update_start_time = time.time()
            current_sequence = _update_sequence_for_next_prediction(
                current_sequence, day_prediction[0], scaler, X, predictions
            )
            logger.debug(f"  Day {day+1} sequence update took {time.time() - update_start_time:.4f}s")
    
    logger.info(f"Multi-step prediction complete. Total time: {time.time() - start_time:.4f}s")
    return predictions

def _generate_ensemble_predictions(models: list, scaler, X: np.ndarray, days: int, _predict_ensemble_func) -> np.ndarray:
    """Generate predictions from each model in the ensemble for confidence intervals"""
    ensemble_predictions = []
    last_sequence = X[-1:, :, :].copy()
    
    logger.info(f"Starting ensemble prediction for confidence intervals for {days} days with {len(models)} models.")
    start_time = time.time() # Add time import
    
    for model_idx, model in enumerate(models):
        model_preds = []
        seq = tf.convert_to_tensor(last_sequence, dtype=tf.float32)
        
        for day in range(days):
            step_start_time = time.time()
            # Use direct model call
            pred = _predict_single_model(model, seq)
            pred_price = _inverse_transform_predictions(pred.numpy().flatten(), scaler)[0]
            model_preds.append(pred_price)
            
            logger.debug(f"  Model {model_idx+1}, Day {day+1} prediction: {pred_price:.2f} (took {time.time() - step_start_time:.4f}s)")
            
            # Update sequence for next prediction if not the last day
            if day < days - 1:
                update_start_time = time.time()
                seq = _update_sequence_for_next_prediction(
                    seq, pred_price, scaler, X, model_preds
                )
                logger.debug(f"  Model {model_idx+1}, Day {day+1} sequence update took {time.time() - update_start_time:.4f}s")
        
        ensemble_predictions.append(model_preds)
    
    logger.info(f"Ensemble prediction for confidence intervals complete. Total time: {time.time() - start_time:.4f}s")
    return np.array(ensemble_predictions)
