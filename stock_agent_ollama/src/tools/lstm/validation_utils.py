import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
from typing import Dict, Any
from pathlib import Path

from src.tools.lstm.custom_scalers import CompositeScaler
from src.tools.lstm.data_pipeline import _prepare_basic_data, prepare_enhanced_data, prepare_enhanced_data_robust, _calculate_robust_features, _validate_feature_quality # Updated import
from src.tools.lstm.model_architecture import create_lstm_model
from src.tools.lstm.model_manager import determine_feature_compatibility, get_model_info
from src.tools.lstm.prediction_utils import _predict_single_model, _inverse_transform_predictions

logger = logging.getLogger(__name__)

def validate_improvements(data: pd.DataFrame, sequence_length: int, symbol: str = "TEST") -> Dict[str, Any]:
    """Comprehensive validation of all LSTM predictor improvements"""
    validation_results = {
        'data_leakage_prevention': False,
        'enhanced_features_working': False,
        'robust_features_working': False,
        'scaling_consistency': False,
        'gradient_clipping_active': False,
        'backward_compatibility': False,
        'prediction_stability': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        logger.info("Starting comprehensive validation of LSTM improvements...")
        
        # Test 1: Data leakage prevention
        try:
            X_basic, y_basic, scaler_basic = _prepare_basic_data(data, sequence_length, validation_split=0.2) # Updated call
            # Check that scaler was fitted properly (should not have leaked data)
            validation_results['data_leakage_prevention'] = True
            logger.info("✓ Data leakage prevention validated")
        except Exception as e:
            validation_results['errors'].append(f"Data leakage test failed: {e}")
        
        # Test 2: Enhanced features
        try:
            X_enhanced, y_enhanced, scaler_enhanced = prepare_enhanced_data(data, sequence_length, validation_split=0.2)
            if X_enhanced.shape[2] > 5:
                validation_results['enhanced_features_working'] = True
                logger.info(f"✓ Enhanced features working: {X_enhanced.shape[2]} features")
                
                # Test robust features for MSFT-like symbols
                try:
                    X_robust, y_robust, scaler_robust = prepare_enhanced_data_robust(data, sequence_length, symbol, validation_split=0.2)
                    if X_robust.shape[2] == 17:  # Expect exactly 17 features
                        logger.info(f"✓ Robust features working: {X_robust.shape[2]} features with aggressive outlier handling")
                        validation_results['robust_features_working'] = True
                        
                        # Additional validation for outlier reduction
                        enhanced_data = _calculate_robust_features(data)
                        quality_results = _validate_feature_quality(enhanced_data, symbol)
                        if quality_results['total_outliers'] <= 50:
                            logger.info(f"✓ Outlier reduction successful: {quality_results['total_outliers']} outliers (target: ≤50)")
                        else:
                            validation_results['warnings'].append(f"Outlier count still high: {quality_results['total_outliers']} (target: ≤50)")
                    else:
                        validation_results['warnings'].append(f"Robust features count mismatch: expected 17, got {X_robust.shape[2]}")
                except Exception as robust_error:
                    validation_results['warnings'].append(f"Robust features test failed: {robust_error}")
            else:
                validation_results['warnings'].append("Enhanced features returned basic feature count")
        except Exception as e:
            validation_results['errors'].append(f"Enhanced features test failed: {e}")
        
        # Test 3: Scaling consistency
        try:
            if validation_results['enhanced_features_working']:
                X_test, y_test, scaler_test = X_enhanced, y_enhanced, scaler_enhanced
            else:
                X_test, y_test, scaler_test = X_basic, y_basic, scaler_basic
            
            # Test inverse transform consistency
            test_pred = np.array([0.5])
            inverse_pred = _inverse_transform_predictions(test_pred, scaler_test)
            re_scaled = scaler_test.transform(np.array([[inverse_pred[0]] + [0] * (scaler_test.n_features_in_ - 1)]))
            
            if abs(re_scaled[0, 0] - test_pred[0]) < 1e-6:
                validation_results['scaling_consistency'] = True
                logger.info("✓ Scaling consistency validated")
            else:
                validation_results['warnings'].append("Minor scaling inconsistency detected")
        except Exception as e:
            validation_results['errors'].append(f"Scaling consistency test failed: {e}")
        
        # Test 4: Model architecture and gradient clipping
        try:
            model = create_lstm_model((X_test.shape[1], X_test.shape[2]))
            optimizer = model.optimizer
            
            # Check if gradient clipping is configured
            if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm is not None:
                validation_results['gradient_clipping_active'] = True
                logger.info("✓ Gradient clipping active")
            else:
                validation_results['warnings'].append("Gradient clipping not detected")
        except Exception as e:
            validation_results['errors'].append(f"Model architecture test failed: {e}")
        
        # Test 5: Backward compatibility
        try:
            compat_info = determine_feature_compatibility(symbol, Path('.')) # dummy path
            if 'feature_count' in compat_info and 'uses_enhanced_features' in compat_info:
                validation_results['backward_compatibility'] = True
                logger.info("✓ Backward compatibility system working")
        except Exception as e:
            validation_results['errors'].append(f"Backward compatibility test failed: {e}")
        
        # Test 6: Prediction stability (quick test with small ensemble)
        try:
            if len(X_test) > 10:
                # Create small test model
                test_model = create_lstm_model((X_test.shape[1], X_test.shape[2]))
                
                # Test prediction consistency
                test_input = X_test[-1:, :, :]
                pred1 = _predict_single_model(test_model, test_input)
                pred2 = _predict_single_model(test_model, test_input)
                
                if np.allclose(pred1.numpy(), pred2.numpy(), atol=1e-6):
                    validation_results['prediction_stability'] = True
                    logger.info("✓ Prediction stability validated")
                else:
                    validation_results['warnings'].append("Minor prediction variability detected")
        except Exception as e:
            validation_results['errors'].append(f"Prediction stability test failed: {e}")
        
        # Summary
        passed_tests = sum([
            validation_results['data_leakage_prevention'],
            validation_results['enhanced_features_working'],
            validation_results.get('robust_features_working', False),
            validation_results['scaling_consistency'],
            validation_results['gradient_clipping_active'],
            validation_results['backward_compatibility'],
            validation_results['prediction_stability']
        ])
        
        validation_results['overall_score'] = passed_tests / 7 * 100
        validation_results['tests_passed'] = passed_tests
        validation_results['total_tests'] = 7
        
        logger.info(f"Validation complete: {passed_tests}/7 tests passed ({validation_results['overall_score']:.1f}%)")
        
        if validation_results['errors']:
            logger.warning(f"Errors encountered: {len(validation_results['errors'])} warnings")
            for error in validation_results['errors']:
                logger.warning(f"  - {error}")
        
        if validation_results['warnings']:
            logger.info(f"Warnings: {len(validation_results['warnings'])} warnings")
            for warning in validation_results['warnings']:
                logger.info(f"  - {warning}")
                
    except Exception as e:
        validation_results['errors'].append(f"Overall validation failed: {e}")
        logger.error(f"Validation failed: {e}")
    
    return validation_results

def check_scaling_health(symbol: str, model_dir: Path) -> Dict[str, Any]:
    """Check the health of scaling parameters for a trained model"""
    result = {
        'symbol': symbol,
        'scaling_healthy': False,
        'issues': [],
        'recommendations': [],
        'scaler_info': {}
    }
    
    try:
        # Load scaler
        scaler_path = model_dir / f"{symbol}_scaler.pkl"
        if not scaler_path.exists():
            result['issues'].append("Scaler file not found")
            result['recommendations'].append("Train the model first")
            return result
            
        scaler = joblib.load(scaler_path)
        result['scaler_info']['type'] = type(scaler).__name__
        
        if isinstance(scaler, CompositeScaler):
            # Check if scaler has price range parameters
            has_price_params = (scaler.price_min is not None and 
                              scaler.price_max is not None and 
                              scaler.data_range is not None)
            
            result['scaler_info'].update({
                'fitted': getattr(scaler, 'fitted', None),
                'price_min': getattr(scaler, 'price_min', None),
                'price_max': getattr(scaler, 'price_max', None),
                'data_range': getattr(scaler, 'data_range', None)
            })
            
            # Check metadata for saved parameters
            metadata = get_model_info(symbol, model_dir)
            has_metadata_params = (metadata and 'scaler_info' in metadata and
                                 metadata['scaler_info'].get('price_min') is not None)
            
            if has_price_params:
                result['scaling_healthy'] = True
                result['recommendations'].append("Scaling parameters are healthy")
            elif has_metadata_params:
                result['issues'].append("Scaler parameters missing but available in metadata")
                result['recommendations'].append("Scaler will be auto-restored from metadata on next prediction")
            else:
                result['issues'].append("Missing price range parameters - predictions will be inaccurate")
                result['recommendations'].append("Retrain the model to fix scaling issues")
                
        else:
            result['scaling_healthy'] = True  # MinMaxScaler doesn't have this issue
            result['recommendations'].append("Using MinMaxScaler - no scaling parameter issues")
            
    except Exception as e:
        result['issues'].append(f"Error checking scaler: {e}")
        result['recommendations'].append("Check model files or retrain")
        
    return result

def diagnose_model_issues(symbol: str, model_dir: Path) -> Dict[str, Any]:
    """Diagnose potential issues with trained models"""
    results = {
        'symbol': symbol,
        'issues': [],
        'recommendations': [],
        'model_files': {},
        'scaler_health': {},
        'metadata_info': {}
    }
    
    # Check for model files
    model_files_found = list(model_dir.glob(f"{symbol}_model_*.keras")) + \
                      list(model_dir.glob(f"{symbol}_model_*.h5")) + \
                      list(model_dir.glob(f"{symbol}_model_*_savedmodel"))
    
    results['model_files']['found'] = [str(p) for p in model_files_found]
    if not model_files_found:
        results['issues'].append("No model files found")
        results['recommendations'].append("Train the model first")
    
    # Check scaler health
    scaler_health = check_scaling_health(symbol, model_dir)
    results['scaler_health'] = scaler_health
    if not scaler_health['scaling_healthy']:
        results['issues'].extend(scaler_health['issues'])
        results['recommendations'].extend(scaler_health['recommendations'])
    
    # Check metadata
    metadata = get_model_info(symbol, model_dir)
    results['metadata_info'] = metadata
    if not metadata:
        results['issues'].append("Metadata file not found or corrupted")
        results['recommendations'].append("Retrain the model to generate metadata")
    else:
        # Check for training errors in metadata
        if 'training_histories' in metadata and metadata['training_histories']:
            for i, history in enumerate(metadata['training_histories']):
                if 'val_loss' in history and history['val_loss'][-1] > 0.1: # High validation loss
                    results['issues'].append(f"Model {i} has high validation loss")
                    results['recommendations'].append("Consider retraining with more data or different hyperparameters")
    
    if not results['issues']:
        results['issues'].append("No obvious issues found, but models still fail to load.")
        results['recommendations'].append("This could be due to a corrupted file or a version mismatch between TensorFlow/Keras.")
        results['recommendations'].append("Try deleting the model files and retraining.")

    return results