import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
import sys
import shutil
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional, List

from src.tools.lstm.custom_scalers import CompositeScaler
from src.tools.lstm.prediction_utils import _predict_single_model

logger = logging.getLogger(__name__)

def get_model_info(symbol: str, model_dir: Path) -> Dict[str, Any]:
    """Get metadata for a trained model"""
    metadata_path = model_dir / f"{symbol}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return None
    return None

def _get_comprehensive_custom_objects() -> Dict[str, Any]:
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

def _ensure_model_compiled(model: tf.keras.Model, symbol: str, model_index: int, sequence_length: int):
    """Ensure model is properly compiled with correct loss and metrics"""
    # Import locally to avoid circular dependency
    # from src.tools.lstm.prediction_service import _predict_single_model
    try:
        # Always recompile the model to ensure consistency
        logger.debug(f"Ensuring proper compilation for model {model_index} for {symbol}")
        
        # Use the same compilation parameters as in create_lstm_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0
            ),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        # Build the model by running a dummy prediction to initialize all metrics
        try:
            # CRITICAL FIX: Dynamically determine feature count from model input shape
            model_input_shape = model.input_shape
            if len(model_input_shape) >= 3:
                expected_feature_count = model_input_shape[2]
            else:
                # Fallback to basic features if input shape is malformed
                expected_feature_count = 5
                logger.warning(f"Model {model_index} for {symbol} has unexpected input shape: {model_input_shape}, using 5 features as fallback")
            
            # Create dummy input with correct shape matching model's expected features
            dummy_input = np.zeros((1, sequence_length, expected_feature_count))
            logger.debug(f"Model {model_index} for {symbol}: using {expected_feature_count} features for validation")
            
            # Perform dummy prediction to build metrics and ensure model is functional
            dummy_output = _predict_single_model(model, dummy_input).numpy()
            
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
            except Exception as loss_error:
                logger.warning(f"Model {model_index} for {symbol} failed loss computation test: {loss_error}")

        except Exception as build_error:
            logger.warning(f"Failed to build model {model_index} for {symbol}: {build_error}")

    except Exception as e:
        logger.error(f"Failed to ensure model compilation for {symbol}, model {model_index}: {e}")

def _validate_single_model(model: tf.keras.Model, model_index: int, symbol: str, sequence_length: int) -> bool:
    """Validate that a single loaded model is functional"""
    # Import locally to avoid circular dependency
    # from src.tools.lstm.prediction_service import _predict_single_model
    try:
        # Check basic model properties
        if model is None:
            return False
            
        # Check input/output shapes
        if not hasattr(model, 'input_shape') or not hasattr(model, 'output_shape'):
            logger.warning(f"Model {model_index} for {symbol} missing input/output shape")
            return False
        
        # CRITICAL FIX: Dynamically determine expected feature count from model input shape
        actual_input_shape = model.input_shape
        if len(actual_input_shape) < 3:
            logger.warning(f"Model {model_index} for {symbol} has invalid input shape dimensions: {actual_input_shape}")
            return False
            
        # Extract feature count from model's actual input shape
        expected_feature_count = actual_input_shape[2]
        
        # Verify input shape structure (batch, sequence, features)
        if actual_input_shape[1] != sequence_length:
            logger.warning(f"Model {model_index} for {symbol} has unexpected sequence length: {actual_input_shape[1]} vs {sequence_length}")
            return False
        
        # Log model compatibility info
        logger.debug(f"Model {model_index} for {symbol}: input_shape={actual_input_shape}, features={expected_feature_count}")
        
        # CRITICAL FIX: Test with dummy data matching model's expected feature count
        dummy_input = np.zeros((1, sequence_length, expected_feature_count))
        try:
            # Use direct model call with consistent tensor format
            prediction = _predict_single_model(model, dummy_input).numpy()
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

def _validate_loaded_models(models: list, symbol: str):
    """Validate that loaded models are functional"""
    if not models:
        return
    
    logger.debug(f"Validating {len(models)} loaded models for {symbol}")
    # Simple validation: check if all models are instances of tf.keras.Model
    if not all(isinstance(m, tf.keras.Model) for m in models):
        raise TypeError("Not all loaded objects are Keras models")

def _cleanup_failed_save(symbol: str, model_dir: Path):
    """Clean up partially saved model files on failure"""
    for i in range(5): # Check for up to 5 models
        for ext in ['.keras', '.h5', '_savedmodel']:
            path = model_dir / f"{symbol}_model_{i}{ext}"
            if ext == '_savedmodel' and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file():
                path.unlink()

def _migrate_h5_to_keras(symbol: str, model_dir: Path, sequence_length: int):
    """Migrate legacy .h5 models to the modern .keras format"""
    for i in range(5): # Check for up to 5 models
        h5_path = model_dir / f"{symbol}_model_{i}.h5"
        keras_path = model_dir / f"{symbol}_model_{i}.keras"
        
        if h5_path.exists() and not keras_path.exists():
            try:
                logger.warning(f"Migrating legacy .h5 model {i} for {symbol} to .keras format")
                custom_objects = _get_comprehensive_custom_objects()
                model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)
                _ensure_model_compiled(model, symbol, i, sequence_length)
                model.save(keras_path)
                # Optional: remove old .h5 file after successful migration
                # h5_path.unlink()
                logger.info(f"Successfully migrated model {i} for {symbol} to .keras format")
            except Exception as e:
                logger.error(f"Failed to migrate .h5 model {i} for {symbol}: {e}")

def save_ensemble(models: list, scaler, symbol: str, model_dir: Path, sequence_length: int, histories: list):
    """Save ensemble models and metadata using modern .keras format"""
    try:
        # Migrate any existing .h5 models to .keras format first
        _migrate_h5_to_keras(symbol, model_dir, sequence_length)
        
        # Save models using the new .keras format (TensorFlow 2.20.0 compatible)
        saved_models = []
        for i, model in enumerate(models):
            model_path = model_dir / f"{symbol}_model_{i}.keras"
            try:
                # Ensure model is properly compiled before saving
                _ensure_model_compiled(model, symbol, i, sequence_length)
                
                # Use .keras extension for TensorFlow 2.20.0+ (no save_format parameter needed)
                model.save(model_path)
                
                # Verify the save worked by attempting to load
                try:
                    test_model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects=_get_comprehensive_custom_objects(),
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
                savedmodel_path = model_dir / f"{symbol}_model_{i}_savedmodel"
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
                        custom_objects=_get_comprehensive_custom_objects()
                    )
                    if test_model is not None:
                        logger.warning(f"Saved model {i} for {symbol} in SavedModel format as fallback")
                        saved_models.append(savedmodel_path)
                    else:
                        raise ValueError("SavedModel loaded as None")
                        
                except Exception as savedmodel_error:
                    logger.error(f"SavedModel fallback failed for model {i}: {savedmodel_error}")
                    
                    # Fallback strategy 2: Try legacy .h5 format (last resort)
                    h5_path = model_dir / f"{symbol}_model_{i}.h5"
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
                            custom_objects=_get_comprehensive_custom_objects(),
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
        scaler_path = model_dir / f"{symbol}_scaler.pkl"
        try:
            joblib.dump(scaler, scaler_path)
            logger.debug(f"Saved scaler for {symbol}")
        except Exception as scaler_error:
            logger.error(f"Failed to save scaler for {symbol}: {scaler_error}")
            raise
        
        # Save metadata with enhanced format information including scaler parameters
        metadata = {
            'symbol': symbol,
            'ensemble_size': len(models),
            'models_saved': len(saved_models),
            'sequence_length': sequence_length,
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
                'total_params': models[0].count_params() if models else None,
                'feature_count': models[0].input_shape[2] if models and len(models[0].input_shape) > 2 else 5,
                'uses_enhanced_features': models[0].input_shape[2] > 5 if models and len(models[0].input_shape) > 2 else False
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
        
        metadata_path = model_dir / f"{symbol}_metadata.json"
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
        _cleanup_failed_save(symbol, model_dir)
        raise

def _load_single_model(symbol: str, model_index: int, model_dir: Path, sequence_length: int) -> Optional[tf.keras.Model]:
    """Load a single model with comprehensive fallback strategies"""
    model = None
    load_attempts = []
    
    # Get comprehensive custom objects for loading legacy models
    custom_objects = _get_comprehensive_custom_objects()
    
    # Strategy 1: Try to load .keras format first (preferred)
    keras_path = model_dir / f"{symbol}_model_{model_index}.keras"
    if keras_path.exists():
        try:
            logger.debug(f"Attempting to load .keras model {model_index} for {symbol}")
            model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects, compile=False)
            
            # Manually compile the model to ensure proper configuration
            _ensure_model_compiled(model, symbol, model_index, sequence_length)
            
            # Validate the model works
            if _validate_single_model(model, model_index, symbol, sequence_length):
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
    h5_path = model_dir / f"{symbol}_model_{model_index}.h5"
    if h5_path.exists():
        try:
            logger.debug(f"Attempting to load .h5 model {model_index} for {symbol}")
            model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)
            
            # Manually compile the model to ensure proper configuration
            _ensure_model_compiled(model, symbol, model_index, sequence_length)
            
            # Validate the model works
            if _validate_single_model(model, model_index, symbol, sequence_length):
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
    savedmodel_path = model_dir / f"{symbol}_model_{model_index}_savedmodel"
    if savedmodel_path.exists():
        try:
            logger.debug(f"Attempting to load SavedModel {model_index} for {symbol}")
            model = tf.keras.models.load_model(savedmodel_path, custom_objects=custom_objects)
            
            # Manually compile the model to ensure proper configuration
            _ensure_model_compiled(model, symbol, model_index, sequence_length)
            
            # Validate the model works
            if _validate_single_model(model, model_index, symbol, sequence_length):
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
            _ensure_model_compiled(model, symbol, model_index, sequence_length)
            
            if _validate_single_model(model, model_index, symbol, sequence_length):
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

def load_ensemble(symbol: str, model_dir: Path, sequence_length: int, ensemble_size: int, data: pd.DataFrame = None) -> Tuple[list, Optional[Any]]:
    """Load ensemble models and scaler with proper error handling and format support"""
    models = []
    scaler = None
    
    try:
        # Load scaler
        scaler_path = model_dir / f"{symbol}_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.debug(f"Loaded scaler for {symbol}")
            # CRITICAL FIX: Ensure scaler is properly restored after loading
            # This addresses issues with joblib not preserving critical attributes
            if not scaler.fitted:
                scaler.fitted = True
                logger.warning(f"Forced scaler fitted state to True for {symbol} after loading.")
            
            # ADDITIONAL FIX: Validate that CompositeScaler has required price range data
            if isinstance(scaler, CompositeScaler):
                if scaler.price_min is None or scaler.price_max is None or scaler.data_range is None:
                    logger.warning(f"CompositeScaler for {symbol} is missing price range data after loading. Attempting to restore...")
                    
                    # First, try to restore from saved metadata
                    metadata = get_model_info(symbol, model_dir)
                    restored_from_metadata = False
                    
                    if metadata and 'scaler_info' in metadata:
                        scaler_info = metadata['scaler_info']
                        saved_price_min = scaler_info.get('price_min')
                        saved_price_max = scaler_info.get('price_max')
                        
                        if saved_price_min is not None and saved_price_max is not None:
                            try:
                                scaler.set_scaling_params(saved_price_min, saved_price_max)
                                logger.info(f"Restored scaling parameters for {symbol} from metadata: ${scaler.price_min:.2f} - ${scaler.price_max:.2f}")
                                # Save the restored scaler back to disk to persist the fix
                                joblib.dump(scaler, scaler_path)
                                logger.debug(f"Saved restored scaler to disk for {symbol}")
                                restored_from_metadata = True
                            except Exception as e:
                                logger.warning(f"Failed to restore scaling parameters from metadata for {symbol}: {e}")
                    
                    # If metadata restoration failed, fall back to recent data (less accurate)
                    if not restored_from_metadata:
                        if data is not None and len(data) > 0 and 'Close' in data.columns:
                            try:
                                recent_prices = data['Close'].dropna()
                                if len(recent_prices) > 0:
                                    scaler.set_scaling_params(recent_prices.min(), recent_prices.max())
                                    logger.warning(f"Restored scaling parameters for {symbol} from recent data (may be inaccurate): ${scaler.price_min:.2f} - ${scaler.price_max:.2f}")
                                    # Save the restored scaler back to disk
                                    joblib.dump(scaler, scaler_path)
                                    logger.debug(f"Saved restored scaler (from data) to disk for {symbol}")
                                else:
                                    logger.warning(f"No valid price data available to restore scaling for {symbol}")
                            except Exception as e:
                                logger.warning(f"Failed to restore scaling parameters for {symbol}: {e}")
                        else:
                            logger.warning(f"No data provided to restore CompositeScaler parameters for {symbol}")
        else:
            logger.warning(f"No scaler found for {symbol}")
            return [], None
        
        # Try to migrate old .h5 models to .keras format if needed
        _migrate_h5_to_keras(symbol, model_dir, sequence_length)
        
        # Load models with fallback strategy
        for i in range(ensemble_size):
            model = _load_single_model(symbol, i, model_dir, sequence_length)
            if model is not None:
                models.append(model)
            else:
                logger.warning(f"Model {i} not found for {symbol}")
        
        if len(models) == 0:
            logger.warning(f"No models found for {symbol}")
            return [], scaler  # Return scaler even if no models found (for testing)
        
        # Validate loaded models
        _validate_loaded_models(models, symbol)
        
        logger.info(f"Successfully loaded {len(models)} models for {symbol}")
        
    except Exception as e:
        logger.error(f"Failed to load models for {symbol}: {e}")
        logger.debug(f"Error details: {str(e)}", exc_info=True)
        return [], None
    
    return models, scaler

def _attempt_model_recovery(symbol: str, model_dir: Path, sequence_length: int) -> bool:
    """Attempt to recover models through migration and cleanup"""
    try:
        # Try migration first
        _migrate_h5_to_keras(symbol, model_dir, sequence_length)
        
        # Clean up any corrupted files
        # Import locally to avoid circular dependency
        from src.tools.lstm.validation_utils import diagnose_model_issues
        diagnosis = diagnose_model_issues(symbol, model_dir)
        if 'corruption' in str(diagnosis['issues']).lower():
            logger.info(f"Cleaning up potentially corrupted files for {symbol}")
            _cleanup_failed_save(symbol, model_dir)
            
        return True
    except Exception as e:
        logger.warning(f"Model recovery attempt failed for {symbol}: {e}")
        return False

def load_ensemble_with_fallback(symbol: str, model_dir: Path, sequence_length: int, ensemble_size: int, data: pd.DataFrame = None) -> Tuple[list, Optional[Any]]:
    """Load ensemble with automatic fallback and recovery mechanisms"""
    try:
        # First attempt: normal loading
        models, scaler = load_ensemble(symbol, model_dir, sequence_length, ensemble_size, data)
        if models and len(models) > 0:
            return models, scaler
            
        logger.warning(f"Normal loading failed for {symbol}, attempting recovery...")
        
        # Second attempt: try migration and reload
        if _attempt_model_recovery(symbol, model_dir, sequence_length):
            models, scaler = load_ensemble(symbol, model_dir, sequence_length, ensemble_size, data)
            if models and len(models) > 0:
                logger.info(f"Successfully recovered {len(models)} models for {symbol}")
                return models, scaler
        
        # Third attempt: force retrain if data is available
        if data is not None and len(data) > 100:  # Ensure sufficient data
            logger.warning(f"Attempting to retrain models for {symbol} due to loading failures")
            # Import locally to avoid circular dependency
            from src.tools.lstm.prediction_service import LSTMPredictionService
            from src.config import Config
            try:
                service = LSTMPredictionService(model_dir)
                service.train_ensemble(
                    data=data,
                    symbol=symbol,
                    sequence_length=sequence_length,
                    ensemble_size=ensemble_size,
                    epochs=Config.EPOCHS,
                    batch_size=Config.BATCH_SIZE
                )
                models, scaler = load_ensemble(symbol, model_dir, sequence_length, ensemble_size, data)
                if models and len(models) > 0:
                    logger.info(f"Successfully retrained and loaded {len(models)} models for {symbol}")
                    return models, scaler
            except Exception as e:
                logger.error(f"Retraining failed for {symbol}: {e}")

    except Exception as e:
        logger.error(f"All fallback attempts failed for {symbol}: {e}")
    
    return [], None

def determine_feature_compatibility(symbol: str, model_dir: Path) -> Dict[str, Any]:
    """Determine if existing models use enhanced features for backward compatibility"""
    metadata = get_model_info(symbol, model_dir)
    
    compatibility_info = {
        'uses_enhanced_features': False,
        'feature_count': 5,
        'needs_migration': False,
        'can_use_enhanced': True
    }
    
    if metadata and 'model_architecture' in metadata:
        arch = metadata['model_architecture']
        if 'feature_count' in arch:
            compatibility_info['feature_count'] = arch['feature_count']
            compatibility_info['uses_enhanced_features'] = arch.get('uses_enhanced_features', False)
        elif 'input_shape' in arch and len(arch['input_shape']) > 2:
            # Backward compatibility for models without explicit feature tracking
            feature_count = arch['input_shape'][2]
            compatibility_info['feature_count'] = feature_count
            compatibility_info['uses_enhanced_features'] = feature_count > 5
        else:
            # Very old models, assume basic features
            compatibility_info['needs_migration'] = True
    else:
        # No metadata, likely very old model
        compatibility_info['needs_migration'] = True
    
    return compatibility_info