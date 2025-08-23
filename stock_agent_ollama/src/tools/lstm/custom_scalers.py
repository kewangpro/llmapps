import numpy as np
import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class CompositeScaler:
    """Composite scaler that maintains backward compatibility with MinMaxScaler interface"""
    
    def __init__(self, feature_names: List[str], symbol: str = "UNKNOWN", validation_split: Optional[float] = None):
        self.feature_names = list(feature_names)
        logger.debug(f"CompositeScaler initialized with feature_names: {self.feature_names}")
        self.symbol = symbol
        self.validation_split = validation_split
        self.n_features_in_ = len(feature_names)
        self.individual_scalers = {}
        self.feature_range = (0, 1)  # For compatibility
        
        # Dynamic scaling parameters - stored from actual training data
        self.price_min = None
        self.price_max = None
        self.data_range = None
        self.data_scale = None
        self.fitted = False
        
    def fit_transform(self, X) -> np.ndarray:
        """Fit and transform data - store comprehensive scaling parameters from actual data"""
        if isinstance(X, pd.DataFrame):
            data_to_fit = X
        elif isinstance(X, np.ndarray):
            data_to_fit = X
        else:
            raise TypeError("Input to fit_transform must be a Pandas DataFrame or NumPy array.")

        scaled_data_array = np.zeros_like(data_to_fit, dtype=np.float32)

        # Fit individual scalers for each feature
        for i, col_name in enumerate(self.feature_names):
            if isinstance(data_to_fit, pd.DataFrame):
                feature_data = data_to_fit[col_name].values.reshape(-1, 1)
            else: # numpy array
                feature_data = data_to_fit[:, i].reshape(-1, 1)
            
            # Create and fit adaptive scaler for this feature
            # This import will be handled by data_pipeline.py
            from src.tools.lstm.data_pipeline import _create_adaptive_scaler 
            scaler = _create_adaptive_scaler(feature_data.flatten(), col_name, self.symbol)
            self.individual_scalers[col_name] = scaler
            logger.debug(f"Fitted individual scaler for {col_name}")
            
            scaled_data_array[:, i] = scaler.fit_transform(feature_data).flatten()

        # Store comprehensive scaling parameters for 'Close' price
        price_data = None
        if 'Close' in self.feature_names:
            close_idx = self.feature_names.index('Close')
            if isinstance(X, pd.DataFrame):
                price_data = X['Close'].dropna()
            else: # numpy array
                price_data = X[:, close_idx]
                price_data = price_data[~np.isnan(price_data)]

        if price_data is not None and len(price_data) > 0:
            self.price_min = float(price_data.min())
            self.price_max = float(price_data.max())
            self.data_range = self.price_max - self.price_min
            self.data_scale = self.data_range / (self.feature_range[1] - self.feature_range[0]) if self.data_range > 0 else 0
            print(f"   📊 Dynamic scaling fitted for {self.symbol}: range ${self.price_min:.2f} - ${self.price_max:.2f}")
            logger.debug(f"CompositeScaler fitted with price_min={self.price_min}, price_max={self.price_max}")
            logger.debug(f"Individual scalers fitted: {list(self.individual_scalers.keys())}")
        
        self.fitted = True
        return scaled_data_array
    
    def transform(self, X) -> np.ndarray:
        """Transform data - scales X using fitted min/max values"""
        if not self.fitted:
            logger.warning("Scaler not fitted, cannot transform data accurately.")
            if isinstance(X, pd.DataFrame):
                return X.values
            return X # Or raise an error, depending on desired behavior

        if isinstance(X, pd.DataFrame):
            data_to_scale = X
        elif isinstance(X, np.ndarray):
            data_to_scale = X
        else:
            raise TypeError("Input to transform must be a Pandas DataFrame or NumPy array.")

        scaled_data_array = np.zeros_like(data_to_scale, dtype=np.float32)
        
        for i, col_name in enumerate(self.feature_names):
            logger.debug(f"Transforming feature: {col_name}")
            if isinstance(data_to_scale, pd.DataFrame):
                feature_data = data_to_scale[col_name].values.reshape(-1, 1)
            else: # numpy array
                feature_data = data_to_scale[:, i].reshape(-1, 1)
            
            # Special handling for 'Close' price: prioritize overall price_min/max
            if col_name == 'Close' and self.price_min is not None and self.data_range is not None and self.data_range > 0:
                scaled_data_array[:, i] = (feature_data.flatten() - self.price_min) / self.data_range
            # Use the individual scaler for other features or if price_min/data_range are not set for 'Close'
            elif col_name in self.individual_scalers:
                scaled_data_array[:, i] = self.individual_scalers[col_name].transform(feature_data).flatten()
            else:
                logger.warning(f"Individual scaler for feature '{col_name}' not found in CompositeScaler. Copying raw data.")
                scaled_data_array[:, i] = feature_data.flatten()
            
        return scaled_data_array
    
    def inverse_transform(self, X):
        """Inverse transform predictions to original scale - simplified for compatibility"""
        # For backward compatibility, assume first feature is Close price
        # Create dummy array for inverse transform
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Create dummy full feature array for compatibility with existing inverse_transform calls
        X_flat = X.flatten()
        dummy = np.zeros((len(X_flat), self.n_features_in_))
        dummy[:, 0] = X_flat
        
        # Dynamic inverse scaling using actual training data statistics
        result = dummy.copy()
        
        # Check if we have dynamic scaling available (with backward compatibility)
        has_dynamic_scaling = (
            hasattr(self, 'fitted') and self.fitted and 
            self.price_min is not None and self.price_max is not None and self.data_range is not None
        )
        
        if has_dynamic_scaling:
            # Use actual price range from training data - most accurate approach
            result[:, 0] = dummy[:, 0] * self.data_range + self.price_min
            
        elif self.price_min is not None and self.price_max is not None:
            # Backward compatibility: Use stored price range even without full dynamic scaling
            price_range = self.price_max - self.price_min
            result[:, 0] = dummy[:, 0] * price_range + self.price_min
            
        else:
            # Final fallback: estimate based on typical price ranges for different symbols
            # This is only used if no price range information is available
            print(f"   ⚠️  Warning: Using intelligent fallback scaling for {self.symbol} (no price range data available)")
            
            # Intelligent fallback based on symbol patterns and typical price ranges
            if self.symbol.startswith(('GOOGL', 'GOOG', 'BRK')):
                # Very high-priced stocks (typically $100-$500+)
                result[:, 0] = dummy[:, 0] * 400 + 100
            elif self.symbol in ['AAPL', 'MSFT', 'NVDA', 'TSLA']:
                # Large-cap tech stocks (typically $150-$600)
                result[:, 0] = dummy[:, 0] * 450 + 150
            elif self.symbol.startswith(('AMZN', 'META')):
                # High-priced growth stocks
                result[:, 0] = dummy[:, 0] * 350 + 100
            elif len(self.symbol) <= 4 and self.symbol.isupper():
                # Standard stocks (most common range $20-$200)
                result[:, 0] = dummy[:, 0] * 180 + 20
            else:
                # Conservative default for unknown symbols
                result[:, 0] = dummy[:, 0] * 100 + 50
                
        return result  # Return 2D array for compatibility
    
    def get_scaling_info(self):
        """Get comprehensive scaling information for debugging/logging"""
        return {
            'symbol': self.symbol,
            'fitted': self.fitted,
            'price_min': self.price_min,
            'price_max': self.price_max,
            'data_range': self.data_range,
            'data_scale': self.data_scale,
            'n_features': self.n_features_in_
        }
    
    def set_scaling_params(self, price_min, price_max):
        """Manually set scaling parameters if needed"""
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.data_range = self.price_max - self.price_min
        if self.data_range > 0:
            self.data_scale = self.data_range / (self.feature_range[1] - self.feature_range[0])
        else:
            self.data_scale = 0
        self.fitted = True
        print(f"   📊 Manual scaling set for {self.symbol}: range ${self.price_min:.2f} - ${self.price_max:.2f}")
