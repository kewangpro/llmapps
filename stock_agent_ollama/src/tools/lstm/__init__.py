"""
LSTM Neural Network System

This module provides a complete LSTM prediction framework:
- Model architecture and ensemble management
- Data pipeline and feature engineering
- Custom scalers for financial data
- Prediction service with validation
- Robust error handling and diagnostics
"""

from src.tools.lstm.prediction_service import LSTMPredictionService
from src.tools.lstm.model_manager import save_ensemble, load_ensemble_with_fallback, get_model_info
from src.tools.lstm.model_architecture import create_lstm_model
from src.tools.lstm.data_pipeline import prepare_enhanced_data, prepare_enhanced_data_robust
from src.tools.lstm.custom_scalers import CompositeScaler
from src.tools.lstm.validation_utils import validate_improvements, check_scaling_health, diagnose_model_issues
from src.tools.lstm.prediction_utils import (
    _predict_single_model,
    _inverse_transform_predictions,
    _generate_multi_step_predictions,
    _generate_ensemble_predictions
)

__all__ = [
    # Main service
    'LSTMPredictionService',

    # Model management
    'save_ensemble',
    'load_ensemble_with_fallback',
    'get_model_info',
    'create_lstm_model',

    # Data preparation
    'prepare_enhanced_data',
    'prepare_enhanced_data_robust',
    'CompositeScaler',

    # Validation
    'validate_improvements',
    'check_scaling_health',
    'diagnose_model_issues',

    # Prediction utilities
    '_predict_single_model',
    '_inverse_transform_predictions',
    '_generate_multi_step_predictions',
    '_generate_ensemble_predictions',
]
