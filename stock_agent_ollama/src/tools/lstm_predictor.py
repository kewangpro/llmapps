import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple

from src.config import Config
from src.tools.lstm.prediction_service import LSTMPredictionService

logger = logging.getLogger(__name__)

class LSTMPredictor:
    """LSTM ensemble for stock price prediction"""
    
    def __init__(self, model_dir: Path = None):
        self.service = LSTMPredictionService(model_dir)

    def prepare_data(self, data: pd.DataFrame) -> Tuple[Any, Any, Any]:
        """Prepare data for LSTM training"""
        # This method is now handled internally by the service's train/predict methods
        # but kept for backward compatibility if external calls exist.
        # It will use the robust enhanced data preparation by default.
        # Note: This method is not directly used by the LSTMPredictionService anymore for data prep
        # It's here for external compatibility if something still calls it.
        # For actual data preparation, refer to data_pipeline.py via LSTMPredictionService.
        raise NotImplementedError("prepare_data is now handled internally by LSTMPredictionService.train_ensemble or .predict")
        
    def train_ensemble(
        self,
        data: pd.DataFrame,
        symbol: str,
        validation_split: float = 0.2,
        epochs: int = None,
        batch_size: int = None,
        progress_callback: Any = None
    ) -> Dict[str, Any]:
        """Train ensemble of LSTM models"""
        return self.service.train_ensemble(
            data=data,
            symbol=symbol,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=progress_callback
        )

    def predict(
        self,
        symbol: str,
        data: pd.DataFrame,
        days: int = None,
        ensemble_size: int = None,
        prediction_callback: Any = None
    ) -> Dict[str, Any]:
        """Generate predictions using trained ensemble with improved error handling

        Args:
            symbol: Stock symbol
            data: Historical price data
            days: Number of days to predict
            ensemble_size: Number of models in ensemble
            prediction_callback: Optional callback for prediction progress updates
        """
        return self.service.predict(
            symbol=symbol,
            data=data,
            days=days,
            ensemble_size=ensemble_size,
            prediction_callback=prediction_callback
        )

    def validate_improvements(self, data: pd.DataFrame, symbol: str = "TEST") -> Dict[str, Any]:
        """Comprehensive validation of all LSTM predictor improvements"""
        return self.service.validate_model(data, symbol)

    def check_scaling_health(self, symbol: str) -> Dict[str, Any]:
        """Check the health of scaling parameters for a trained model"""
        return self.service.check_scaling_health(symbol)

    def diagnose_model_issues(self, symbol: str) -> Dict[str, Any]:
        """Diagnose potential issues with trained models"""
        return self.service.diagnose_model_issues(symbol)

    def get_model_info(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a trained model"""
        return self.service.get_model_info(symbol)

    def is_model_trained(self, symbol: str) -> bool:
        """Check if a model is trained for a given symbol"""
        return self.service.get_model_info(symbol) is not None # FIX: Call through service

    def force_retrain_if_broken(self, symbol: str, data: pd.DataFrame) -> bool:
        """Force retrain if model is broken"""
        return self.service.force_retrain_if_broken(symbol, data)
