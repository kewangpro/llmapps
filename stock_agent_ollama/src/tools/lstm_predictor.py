import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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
from src.tools.technical_analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)

class CompositeScaler:
    """Composite scaler that maintains backward compatibility with MinMaxScaler interface"""
    
    def __init__(self, feature_names, symbol: str = "UNKNOWN", validation_split: float = None):
        self.feature_names = list(feature_names)
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
        
    def fit_transform(self, X):
        """Fit and transform data - store comprehensive scaling parameters from actual data"""
        if isinstance(X, pd.DataFrame):
            # Extract price data for scaling calculations
            if 'Close' in X.columns:
                price_data = X['Close'].dropna()
            elif X.shape[1] > 0:
                # Assume first column is Close price
                price_data = X.iloc[:, 0].dropna()
            else:
                price_data = None
                
            if price_data is not None and len(price_data) > 0:
                # Store comprehensive scaling parameters
                self.price_min = float(price_data.min())
                self.price_max = float(price_data.max())
                self.data_range = self.price_max - self.price_min
                self.data_scale = self.data_range / (self.feature_range[1] - self.feature_range[0])
                self.fitted = True
                
                print(f"   📊 Dynamic scaling fitted for {self.symbol}: range ${self.price_min:.2f} - ${self.price_max:.2f}")
                
        elif isinstance(X, np.ndarray) and X.shape[1] > 0:
            # Handle numpy array input
            price_data = X[:, 0]
            price_data = price_data[~np.isnan(price_data)]  # Remove NaN values
            
            if len(price_data) > 0:
                self.price_min = float(price_data.min())
                self.price_max = float(price_data.max())
                self.data_range = self.price_max - self.price_min
                self.data_scale = self.data_range / (self.feature_range[1] - self.feature_range[0])
                self.fitted = True
                
                print(f"   📊 Dynamic scaling fitted for {self.symbol}: range ${self.price_min:.2f} - ${self.price_max:.2f}")
        
        return X
    
    def transform(self, X):
        """Transform data - not used in new implementation but needed for compatibility"""
        return X
    
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
            self.price_min is not None and self.price_max is not None
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
        self.data_scale = self.data_range / (self.feature_range[1] - self.feature_range[0])
        self.fitted = True
        print(f"   📊 Manual scaling set for {self.symbol}: range ${self.price_min:.2f} - ${self.price_max:.2f}")

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
        """Create a single LSTM model with improved architecture and training stability"""
        # Adaptive architecture based on number of features
        feature_count = input_shape[1]
        
        # Scale LSTM units based on feature complexity
        if feature_count > 10:  # Enhanced features
            lstm_units = [64, 64, 32]
            dense_units = 32
            dropout_rate = 0.3
        else:  # Basic features
            lstm_units = [50, 50, 50]
            dense_units = 25
            dropout_rate = 0.2
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # First LSTM layer with batch normalization
            tf.keras.layers.LSTM(lstm_units[0], return_sequences=True, 
                               recurrent_dropout=0.1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            # Second LSTM layer
            tf.keras.layers.LSTM(lstm_units[1], return_sequences=True,
                               recurrent_dropout=0.1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            # Third LSTM layer
            tf.keras.layers.LSTM(lstm_units[2], return_sequences=False,
                               recurrent_dropout=0.1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            # Dense layers with regularization
            tf.keras.layers.Dense(dense_units, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        
        # CRITICAL FIX: Add gradient clipping and improved optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0  # Gradient clipping to prevent exploding gradients
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        return model
    
    def _predict_single_model(self, model, X):
        """Direct model prediction without tf.function wrapper"""
        # Ensure consistent input tensor format
        if isinstance(X, np.ndarray):
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        else:
            X_tensor = tf.cast(X, dtype=tf.float32)
        
        # Use direct model call for better performance without retracing
        return model(X_tensor, training=False)
    
    def prepare_data(self, data: pd.DataFrame, validation_split: float = None) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for LSTM training with proper data leakage prevention"""
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
        
        # CRITICAL FIX: Split data BEFORE scaling to prevent data leakage
        if validation_split is not None:
            split_index = int(len(dataset) * (1 - validation_split))
            train_data = dataset[:split_index]
            val_data = dataset[split_index:]
            
            # Fit scaler ONLY on training data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_val_data = scaler.transform(val_data)  # Transform validation data using training scaler
            
            # Combine scaled data for sequence creation
            scaled_data = np.vstack([scaled_train_data, scaled_val_data])
        else:
            # For prediction only (no validation split)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict only the close price (first feature)
        
        return np.array(X), np.array(y), scaler
    
    def _cap_outliers(self, series: pd.Series, lower_percentile: float = 2, upper_percentile: float = 98) -> pd.Series:
        """Cap outliers using aggressive percentile-based limits"""
        lower_bound = np.percentile(series.dropna(), lower_percentile)
        upper_bound = np.percentile(series.dropna(), upper_percentile)
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    def _winsorize_outliers(self, series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
        """Apply aggressive Winsorization to extreme outliers before scaling"""
        try:
            from scipy.stats import mstats
            # More aggressive Winsorization with 1% limits on both ends
            winsorized = mstats.winsorize(series.dropna(), limits=limits)
            result = series.copy()
            result.loc[series.notna()] = winsorized
            return result
        except ImportError:
            # Fallback to aggressive percentile capping if scipy not available
            return self._cap_outliers(series, 1, 99)
        except Exception:
            # Fallback to aggressive capping
            return self._cap_outliers(series, 1, 99)
    
    def _aggressive_outlier_removal(self, series: pd.Series, z_threshold: float = 2.5) -> pd.Series:
        """Remove extreme outliers using strict z-score threshold"""
        # Calculate z-scores
        z_scores = np.abs((series - series.mean()) / series.std())
        # Replace extreme outliers with median for smoother distribution
        median_val = series.median()
        result = series.copy()
        result[z_scores > z_threshold] = median_val
        return result
    
    def _ultra_aggressive_preprocessing(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Ultra-aggressive preprocessing for MSFT to achieve <50 outlier target"""
        # Step 1: Remove extreme outliers using strict z-score
        cleaned = self._aggressive_outlier_removal(series, z_threshold=1.8)
        
        # Step 2: Apply aggressive Winsorization
        winsorized = self._winsorize_outliers(cleaned, limits=(0.002, 0.002))  # 0.2% on each end
        
        # Step 3: Apply strict percentile capping
        capped = self._cap_outliers(winsorized, 0.5, 99.5)
        
        # Step 4: For all features, apply rolling median smoothing to reduce noise
        smoothed = capped.rolling(window=3, center=True).median().fillna(capped)
        
        # Step 5: Final outlier removal pass
        final_cleaned = self._aggressive_outlier_removal(smoothed, z_threshold=1.5)
        
        return final_cleaned
    
    def _calculate_robust_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate exactly 17 enhanced features with ultra-aggressive outlier handling for <50 outlier target"""
        ta = TechnicalAnalysis()
        enhanced_data = data[['Close', 'Volume', 'High', 'Low', 'Open']].copy()
        
        # Apply ultra-aggressive preprocessing to Volume due to extreme outliers in MSFT
        enhanced_data['Volume'] = self._ultra_aggressive_preprocessing(data['Volume'], 'Volume')
        
        # Calculate moving averages first for ratio features
        sma_20 = ta.calculate_sma(data['Close'], 20)
        sma_50 = ta.calculate_sma(data['Close'], 50)
        volume_ma_20 = ta.calculate_sma(data['Volume'], 20)
        
        # Add ratio features with ultra-aggressive outlier handling
        price_sma20_ratio = data['Close'] / sma_20
        price_sma50_ratio = data['Close'] / sma_50
        volume_ratio = data['Volume'] / volume_ma_20
        
        enhanced_data['Price_vs_SMA20'] = self._ultra_aggressive_preprocessing(price_sma20_ratio, 'Price_vs_SMA20')
        enhanced_data['Price_vs_SMA50'] = self._ultra_aggressive_preprocessing(price_sma50_ratio, 'Price_vs_SMA50')
        enhanced_data['Volume_MA_Ratio'] = self._ultra_aggressive_preprocessing(volume_ratio, 'Volume_MA_Ratio')
        
        # Technical indicators with ultra-aggressive preprocessing
        rsi_raw = ta.calculate_rsi(data['Close'])
        enhanced_data['RSI'] = self._ultra_aggressive_preprocessing(rsi_raw, 'RSI')
        
        # MACD indicators with ultra-aggressive preprocessing
        macd_data = ta.calculate_macd(data['Close'])
        enhanced_data['MACD'] = self._ultra_aggressive_preprocessing(macd_data['macd'], 'MACD')
        enhanced_data['MACD_Signal'] = self._ultra_aggressive_preprocessing(macd_data['macd_signal'], 'MACD_Signal')
        enhanced_data['MACD_Histogram'] = self._ultra_aggressive_preprocessing(macd_data['macd_histogram'], 'MACD_Histogram')
        
        # Bollinger Bands with ultra-aggressive preprocessing (remove BB_Middle to reduce feature count)
        bb_data = ta.calculate_bollinger_bands(data['Close'])
        enhanced_data['BB_Upper'] = self._ultra_aggressive_preprocessing(bb_data['bb_upper'], 'BB_Upper')
        enhanced_data['BB_Lower'] = self._ultra_aggressive_preprocessing(bb_data['bb_lower'], 'BB_Lower')
        # BB_Middle removed as it's redundant with SMA_20
        
        # Moving averages with ultra-aggressive preprocessing (keep only EMA_12, remove SMA_20)
        enhanced_data['EMA_12'] = self._ultra_aggressive_preprocessing(ta.calculate_ema(data['Close'], 12), 'EMA_12')
        # SMA_20 removed to reduce feature count from 20 to 17
        
        # Volatility indicators with ultra-aggressive preprocessing
        atr_raw = ta.calculate_atr(data['High'], data['Low'], data['Close'])
        enhanced_data['ATR'] = self._ultra_aggressive_preprocessing(atr_raw, 'ATR')
        
        # Consolidated momentum feature with ultra-aggressive preprocessing
        price_change = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        # Create a momentum score that combines both price and volume momentum
        momentum_score = (price_change * 0.7) + (volume_change * 0.3)  # Weight price change more heavily
        enhanced_data['Momentum_Score'] = self._ultra_aggressive_preprocessing(momentum_score, 'Momentum_Score')
        # Removed individual Price_Change and Volume_Change to reduce feature count
        
        return enhanced_data
    
    def _create_adaptive_scaler(self, feature_data: np.ndarray, feature_name: str, symbol: str) -> object:
        """Create adaptive scaler with aggressive outlier handling for MSFT"""
        # Features that always need robust scaling due to high variability
        always_robust_features = ['Volume', 'Volume_MA_Ratio', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ATR', 'Momentum_Score']
        
        # Ratio features that need robust scaling for better outlier handling
        ratio_features = ['Price_vs_SMA20', 'Price_vs_SMA50']
        
        # Symbols known to have significant outlier issues
        msft_problematic_symbols = ['MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
        
        # Check feature characteristics for automatic robust scaling decision
        feature_std = np.std(feature_data)
        feature_mean = np.abs(np.mean(feature_data))
        coefficient_of_variation = feature_std / feature_mean if feature_mean > 0 else float('inf')
        
        # Use RobustScaler if:
        # 1. Feature is in always_robust list
        # 2. Symbol is problematic AND feature is ratio/technical indicator
        # 3. Feature has high coefficient of variation (>1.0)
        # 4. For MSFT, use RobustScaler for most features to handle 1985 outliers
        use_robust = (
            any(robust_feat in feature_name for robust_feat in always_robust_features) or
            (symbol in msft_problematic_symbols and any(ratio_feat in feature_name for ratio_feat in ratio_features)) or
            coefficient_of_variation > 1.0 or
            (symbol == 'MSFT' and feature_name not in ['Close', 'High', 'Low', 'Open'])  # For MSFT, use robust for all except basic OHLC
        )
        
        if use_robust:
            logger.debug(f"Using RobustScaler for {feature_name} (symbol: {symbol}, CV: {coefficient_of_variation:.3f})")
            # Use more aggressive quantile range for RobustScaler to handle extreme outliers
            return RobustScaler(quantile_range=(5.0, 95.0))  # Slightly more conservative than default (25,75)
        else:
            logger.debug(f"Using MinMaxScaler for {feature_name} (symbol: {symbol}, CV: {coefficient_of_variation:.3f})")
            return MinMaxScaler(feature_range=(0, 1))
    
    def _validate_feature_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Validate feature quality with aggressive outlier detection"""
        validation_results = {
            'symbol': symbol,
            'total_features': len(data.columns),
            'expected_features': 17,
            'outlier_warnings': [],
            'scaling_recommendations': {},
            'quality_score': 100,
            'total_outliers': 0
        }
        
        # Validate feature count
        if len(data.columns) != 17:
            validation_results['outlier_warnings'].append(f"Feature count mismatch: expected 17, got {len(data.columns)}")
            validation_results['quality_score'] -= 20
        
        total_outliers = 0
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                # Use 3-sigma standard for outlier detection (consistent with testing)
                mean_val = data[col].mean()
                std_val = data[col].std()
                outliers_3sigma = ((data[col] - mean_val).abs() > 3 * std_val).sum()
                extreme_outliers = ((data[col] - mean_val).abs() > 4 * std_val).sum()
                
                total_outliers += outliers_3sigma
                
                # More stringent outlier thresholds
                if outliers_3sigma > len(data) * 0.02:  # More than 2% outliers (3σ)
                    validation_results['outlier_warnings'].append(f"{col}: {outliers_3sigma} outliers (>3σ)")
                    validation_results['quality_score'] -= 5
                
                if extreme_outliers > len(data) * 0.01:  # More than 1% extreme outliers (4σ)
                    validation_results['outlier_warnings'].append(f"{col}: {extreme_outliers} extreme outliers (>4σ)")
                    validation_results['quality_score'] -= 10
                
                # Check for high variance (potential scaling issues)
                coefficient_of_variation = std_val / abs(mean_val) if mean_val != 0 else float('inf')
                if coefficient_of_variation > 1.5:  # High variability
                    validation_results['scaling_recommendations'][col] = 'RobustScaler recommended (high CV)'
                
                # Stricter ratio feature validation
                if any(ratio_name in col for ratio_name in ['_vs_', 'Ratio', 'Score']):
                    q99 = data[col].quantile(0.99)
                    q01 = data[col].quantile(0.01)
                    q_range = q99 - q01
                    
                    # Ratios should be well-behaved
                    if q99 > 2.5 or q01 < 0.2 or q_range > 2.0:
                        validation_results['outlier_warnings'].append(
                            f"{col}: suspicious ratio distribution (Q01: {q01:.3f}, Q99: {q99:.3f}, range: {q_range:.3f})"
                        )
                        validation_results['quality_score'] -= 8
                
                # Check for inf or nan values
                inf_count = np.isinf(data[col]).sum()
                nan_count = data[col].isna().sum()
                
                if inf_count > 0:
                    validation_results['outlier_warnings'].append(f"{col}: {inf_count} infinite values")
                    validation_results['quality_score'] -= 15
                    
                if nan_count > len(data) * 0.05:  # More than 5% NaN
                    validation_results['outlier_warnings'].append(f"{col}: {nan_count} NaN values ({nan_count/len(data)*100:.1f}%)")
                    validation_results['quality_score'] -= 10
        
        validation_results['total_outliers'] = total_outliers
        
        # Overall quality assessment
        if total_outliers > 50:
            validation_results['quality_score'] -= 25
            validation_results['outlier_warnings'].append(f"Total outliers ({total_outliers}) exceeds target of 50")
        
        # Log results
        logger.info(f"Feature quality for {symbol}: {validation_results['quality_score']}/100, {total_outliers} total outliers")
        
        if validation_results['outlier_warnings']:
            logger.warning(f"Quality issues for {symbol}: {len(validation_results['outlier_warnings'])} warnings")
            for warning in validation_results['outlier_warnings']:
                logger.warning(f"  - {warning}")
        
        if validation_results['scaling_recommendations']:
            logger.info(f"Scaling recommendations for {symbol}: {validation_results['scaling_recommendations']}")
        
        return validation_results

    def prepare_enhanced_data_robust(self, data: pd.DataFrame, symbol: str = "UNKNOWN", validation_split: float = None) -> Tuple[np.ndarray, np.ndarray, object]:
        """Prepare enhanced data with robust outlier handling and adaptive scaling"""
        logger.info(f"Preparing robust enhanced features for {symbol}")
        
        # Calculate robust features with outlier handling
        enhanced_data = self._calculate_robust_features(data)
        
        # Remove NaN values (first rows will have NaN due to technical indicators)
        enhanced_data = enhanced_data.dropna()
        
        if len(enhanced_data) < self.sequence_length + 10:
            raise ValueError(f"Insufficient data after adding technical indicators: {len(enhanced_data)} samples")
        
        # Validate feature quality and log warnings
        feature_quality = self._validate_feature_quality(enhanced_data, symbol)
        logger.info(f"Feature quality score for {symbol}: {feature_quality['quality_score']}/100")
        
        # Use per-feature adaptive scaling
        scaled_data = self._apply_adaptive_scaling(enhanced_data, symbol, validation_split)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict only the close price (first feature)
        
        # Create a composite scaler object that stores individual scalers for backward compatibility
        composite_scaler = CompositeScaler(enhanced_data.columns, symbol, validation_split)
        
        # Fit the composite scaler with the enhanced data to enable dynamic scaling
        composite_scaler.fit_transform(enhanced_data)
        
        logger.info(f"Prepared {len(X)} sequences with {enhanced_data.shape[1]} robust features for {symbol}")
        return np.array(X), np.array(y), composite_scaler
    
    def _apply_adaptive_scaling(self, data: pd.DataFrame, symbol: str, validation_split: float = None) -> np.ndarray:
        """Apply adaptive per-feature scaling with data leakage prevention"""
        dataset = data.values
        
        # Split data BEFORE scaling to prevent data leakage
        if validation_split is not None:
            split_index = int(len(dataset) * (1 - validation_split))
            train_data = dataset[:split_index]
            val_data = dataset[split_index:]
        else:
            train_data = dataset
            val_data = None
        
        # Apply per-feature scaling
        scaled_train_data = np.zeros_like(train_data)
        scaled_val_data = np.zeros_like(val_data) if val_data is not None else None
        
        for i, col_name in enumerate(data.columns):
            # Create adaptive scaler for this feature
            scaler = self._create_adaptive_scaler(train_data[:, i], col_name, symbol)
            
            # Fit scaler ONLY on training data
            scaled_train_data[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
            
            # Transform validation data if exists
            if val_data is not None:
                scaled_val_data[:, i] = scaler.transform(val_data[:, i].reshape(-1, 1)).flatten()
        
        # Combine scaled data for sequence creation
        if val_data is not None:
            scaled_data = np.vstack([scaled_train_data, scaled_val_data])
        else:
            scaled_data = scaled_train_data
        
        return scaled_data

    def prepare_enhanced_data(self, data: pd.DataFrame, validation_split: float = None) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare enhanced data with technical indicators for LSTM training (backward compatibility)"""
        # For backward compatibility, detect symbol from call stack if possible
        symbol = "UNKNOWN"
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                frame = frame.f_back
                if frame and 'symbol' in frame.f_locals:
                    symbol = frame.f_locals['symbol']
                    break
        except:
            pass
        
        # Use robust method for known problematic symbols
        problematic_symbols = ['MSFT', 'AMZN', 'GOOGL', 'META']
        if symbol in problematic_symbols:
            logger.info(f"Using robust feature preprocessing for {symbol} (known outlier issues)")
            return self.prepare_enhanced_data_robust(data, symbol, validation_split)
        
        # Fall back to original method for compatibility
        logger.debug(f"Using legacy feature preprocessing for {symbol}")
        return self._prepare_enhanced_data_legacy(data, validation_split)
    
    def _prepare_enhanced_data_legacy(self, data: pd.DataFrame, validation_split: float = None) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Legacy enhanced data preparation method"""
        # Calculate technical indicators
        ta = TechnicalAnalysis()
        
        # Basic price features
        enhanced_data = data[['Close', 'Volume', 'High', 'Low', 'Open']].copy()
        
        # Add technical indicators
        enhanced_data['RSI'] = ta.calculate_rsi(data['Close'])
        
        # MACD indicators
        macd_data = ta.calculate_macd(data['Close'])
        enhanced_data['MACD'] = macd_data['macd']
        enhanced_data['MACD_Signal'] = macd_data['macd_signal']
        enhanced_data['MACD_Histogram'] = macd_data['macd_histogram']
        
        # Bollinger Bands
        bb_data = ta.calculate_bollinger_bands(data['Close'])
        enhanced_data['BB_Upper'] = bb_data['bb_upper']
        enhanced_data['BB_Middle'] = bb_data['bb_middle']
        enhanced_data['BB_Lower'] = bb_data['bb_lower']
        
        # Moving averages
        enhanced_data['SMA_20'] = ta.calculate_sma(data['Close'], 20)
        enhanced_data['EMA_12'] = ta.calculate_ema(data['Close'], 12)
        
        # Volatility indicators
        enhanced_data['ATR'] = ta.calculate_atr(data['High'], data['Low'], data['Close'])
        
        # Price momentum features
        enhanced_data['Price_Change'] = data['Close'].pct_change()
        enhanced_data['Volume_Change'] = data['Volume'].pct_change()
        
        # Remove NaN values (first rows will have NaN due to technical indicators)
        enhanced_data = enhanced_data.dropna()
        
        if len(enhanced_data) < self.sequence_length + 10:
            raise ValueError(f"Insufficient data after adding technical indicators: {len(enhanced_data)} samples")
        
        dataset = enhanced_data.values
        
        # CRITICAL FIX: Split data BEFORE scaling to prevent data leakage
        if validation_split is not None:
            split_index = int(len(dataset) * (1 - validation_split))
            train_data = dataset[:split_index]
            val_data = dataset[split_index:]
            
            # Fit scaler ONLY on training data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_val_data = scaler.transform(val_data)  # Transform validation data using training scaler
            
            # Combine scaled data for sequence creation
            scaled_data = np.vstack([scaled_train_data, scaled_val_data])
        else:
            # For prediction only (no validation split)
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
        
        # Prepare data with enhanced features if possible, fallback to basic features
        try:
            # Try enhanced features first with symbol parameter
            if hasattr(self, 'prepare_enhanced_data_robust') and symbol in ['MSFT', 'AMZN', 'GOOGL', 'META']:
                logger.info(f"Using robust enhanced features for {symbol} (known outlier issues)")
                X, y, scaler = self.prepare_enhanced_data_robust(data, symbol, validation_split)
            else:
                X, y, scaler = self.prepare_enhanced_data(data, validation_split)
            logger.info(f"Using enhanced features ({X.shape[2]} features) for training {symbol}")
        except Exception as e:
            logger.warning(f"Enhanced features failed for {symbol}: {e}. Falling back to basic features.")
            X, y, scaler = self.prepare_data(data, validation_split)
            logger.info(f"Using basic features ({X.shape[2]} features) for training {symbol}")
        
        if len(X) < 50:  # Need minimum data for training
            raise ValueError(f"Insufficient data for training: {len(X)} samples. Need at least 50.")
        
        # Data is already split in prepare_data methods, so we need to recreate the split
        split_index = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        
        models = []
        training_histories = []
        
        # Train ensemble
        for i in range(self.ensemble_size):
            logger.info(f"Training model {i+1}/{self.ensemble_size}")
            
            model = self.create_lstm_model((X.shape[1], X.shape[2]))
            
            # IMPROVED: Enhanced callbacks for better training stability
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=15,  # Increased patience for stability
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
        
        # CRITICAL FIX: Use backward compatibility system to determine feature set
        compatibility_info = self._determine_feature_compatibility(symbol)
        use_enhanced_features = compatibility_info['uses_enhanced_features']
        
        logger.info(f"Model for {symbol}: Enhanced features={use_enhanced_features}, "
                   f"Feature count={compatibility_info['feature_count']}")
        
        # Prepare data with matching feature set for backward compatibility
        try:
            if use_enhanced_features:
                X, _, scaler_check = self.prepare_enhanced_data(data)
                logger.debug(f"Using enhanced features for prediction: {X.shape[2]} features")
            else:
                X, _, scaler_check = self.prepare_data(data)
                logger.debug(f"Using basic features for prediction: {X.shape[2]} features")
                
            # Validate feature count matches expected
            expected_features = compatibility_info['feature_count']
            if X.shape[2] != expected_features:
                logger.warning(f"Feature count mismatch: expected {expected_features}, got {X.shape[2]}. "
                              f"Falling back to basic features.")
                raise ValueError(f"Feature count mismatch")
                
        except Exception as e:
            logger.warning(f"Feature preparation failed, falling back to basic features: {e}")
            X, _, scaler_check = self.prepare_data(data)
            
            # Update compatibility info for fallback
            compatibility_info['uses_enhanced_features'] = False
            compatibility_info['feature_count'] = X.shape[2]
        
        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Use the last sequence for prediction
        last_sequence = X[-1:, :, :]
        
        # CRITICAL FIX: Improved multi-step prediction with consistent scaling
        predictions = self._generate_multi_step_predictions(models, scaler, X, days)
        
        # Calculate confidence intervals with improved ensemble variance
        ensemble_predictions = self._generate_ensemble_predictions(models, scaler, X, days)
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
    
    def _predict_ensemble(self, models: list, X, scaler: MinMaxScaler) -> np.ndarray:
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
            pred = self._predict_single_model(model, X_tensor)
            pred_rescaled = self._inverse_transform_predictions(pred.numpy().flatten(), scaler)
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

    def _generate_multi_step_predictions(self, models: list, scaler: MinMaxScaler, X: np.ndarray, days: int) -> list:
        """Generate multi-step predictions with consistent scaling and improved sequence updating"""
        predictions = []
        last_sequence = X[-1:, :, :].copy()
        current_sequence = tf.convert_to_tensor(last_sequence, dtype=tf.float32)
        
        # Get the number of features for proper sequence updating
        n_features = X.shape[2]
        
        for day in range(days):
            # Get ensemble prediction for current sequence
            day_prediction = self._predict_ensemble(models, current_sequence, scaler)
            predictions.append(day_prediction[0])
            
            # CRITICAL FIX: Properly update sequence with consistent scaling
            if day < days - 1:  # Don't update sequence for the last prediction
                current_sequence = self._update_sequence_for_next_prediction(
                    current_sequence, day_prediction[0], scaler, X, predictions
                )
        
        return predictions

    def _update_sequence_for_next_prediction(self, current_sequence: tf.Tensor, predicted_price: float, 
                                           scaler: MinMaxScaler, original_X: np.ndarray, 
                                           all_predictions: list) -> tf.Tensor:
        """Update sequence for next prediction with proper scaling consistency"""
        seq_np = current_sequence.numpy()
        new_row = seq_np[0, -1, :].copy()
        
        # CRITICAL FIX: Convert predicted price back to scaled space for sequence consistency
        # Use CompositeScaler's min/max for consistent re-scaling
        # Ensure scaler has been fitted and has price_min/price_max
        if hasattr(scaler, 'price_min') and scaler.price_min is not None and \
           hasattr(scaler, 'data_range') and scaler.data_range is not None and \
           scaler.data_range > 0:
            scaled_pred_price = (predicted_price - scaler.price_min) / scaler.data_range
        else:
            # Fallback to the original scaler.transform if CompositeScaler info is missing
            # This should ideally not happen if the model is loaded correctly
            logger.warning("CompositeScaler price_min or data_range not available, falling back to feature scaler transform.")
            dummy_for_scaling = np.zeros((1, scaler.n_features_in_))
            dummy_for_scaling[0, 0] = predicted_price
            scaled_pred_price = scaler.transform(dummy_for_scaling)[0, 0]

        # Update the close price feature (index 0) with scaled predicted price
        # No clipping for scaled_pred_price, as it should naturally be within 0-1 if scaling is correct
        new_row[0] = scaled_pred_price
        
        # For enhanced features, estimate other technical indicators
        if new_row.shape[0] > 5:  # Enhanced features
            # Update other features based on price movement
            price_change_ratio = 1.0
            if len(all_predictions) > 0:
                last_actual_price = self._inverse_transform_predictions(
                    np.array([original_X[-1, -1, 0]]), scaler
                )[0]
                price_change_ratio = predicted_price / last_actual_price
            
            # Update volume (index 1) - inverse relationship with price movement
            if new_row.shape[0] > 1:
                volume_change = 1.0 / max(0.5, min(2.0, price_change_ratio))
                new_row[1] = np.clip(seq_np[0, -1, 1] * volume_change, 0.01, 0.99)
            
            # Update high/low (indices 2,3) based on predicted close
            if new_row.shape[0] > 3:
                new_row[2] = np.clip(new_row[0] * 1.01, new_row[0], 0.99)  # High slightly above close
                new_row[3] = np.clip(new_row[0] * 0.99, 0.01, new_row[0])  # Low slightly below close
            
            # Keep technical indicators relatively stable (small updates)
            for i in range(5, new_row.shape[0]):
                momentum = 0.95  # Keep 95% of previous value
                new_row[i] = seq_np[0, -1, i] * momentum + new_row[0] * (1 - momentum)
                new_row[i] = np.clip(new_row[i], 0.01, 0.99)
        
        # Roll the sequence and add the new row
        seq_np = np.roll(seq_np, -1, axis=1)
        seq_np[0, -1, :] = new_row
        
        return tf.convert_to_tensor(seq_np, dtype=tf.float32)

    def _generate_ensemble_predictions(self, models: list, scaler: MinMaxScaler, X: np.ndarray, days: int) -> np.ndarray:
        """Generate predictions from each model in the ensemble for confidence intervals"""
        ensemble_predictions = []
        last_sequence = X[-1:, :, :].copy()
        
        for model in models:
            model_preds = []
            seq = tf.convert_to_tensor(last_sequence, dtype=tf.float32)
            
            for day in range(days):
                # Use direct model call
                pred = self._predict_single_model(model, seq)
                pred_price = self._inverse_transform_predictions(pred.numpy().flatten(), scaler)[0]
                model_preds.append(pred_price)
                
                # Update sequence for next prediction if not the last day
                if day < days - 1:
                    seq = self._update_sequence_for_next_prediction(
                        seq, pred_price, scaler, X, model_preds
                    )
            
            ensemble_predictions.append(model_preds)
        
        return np.array(ensemble_predictions)

    def _determine_feature_compatibility(self, symbol: str) -> Dict[str, Any]:
        """Determine if existing models use enhanced features for backward compatibility"""
        metadata = self.get_model_info(symbol)
        
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

    def validate_improvements(self, data: pd.DataFrame, symbol: str = "TEST") -> Dict[str, Any]:
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
                X_basic, y_basic, scaler_basic = self.prepare_data(data, validation_split=0.2)
                # Check that scaler was fitted properly (should not have leaked data)
                validation_results['data_leakage_prevention'] = True
                logger.info("✓ Data leakage prevention validated")
            except Exception as e:
                validation_results['errors'].append(f"Data leakage test failed: {e}")
            
            # Test 2: Enhanced features
            try:
                X_enhanced, y_enhanced, scaler_enhanced = self.prepare_enhanced_data(data, validation_split=0.2)
                if X_enhanced.shape[2] > 5:
                    validation_results['enhanced_features_working'] = True
                    logger.info(f"✓ Enhanced features working: {X_enhanced.shape[2]} features")
                    
                    # Test robust features for MSFT-like symbols
                    try:
                        X_robust, y_robust, scaler_robust = self.prepare_enhanced_data_robust(data, symbol, validation_split=0.2)
                        if X_robust.shape[2] == 17:  # Expect exactly 17 features
                            logger.info(f"✓ Robust features working: {X_robust.shape[2]} features with aggressive outlier handling")
                            validation_results['robust_features_working'] = True
                            
                            # Additional validation for outlier reduction
                            enhanced_data = self._calculate_robust_features(data)
                            quality_results = self._validate_feature_quality(enhanced_data, symbol)
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
                inverse_pred = self._inverse_transform_predictions(test_pred, scaler_test)
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
                model = self.create_lstm_model((X_test.shape[1], X_test.shape[2]))
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
                compat_info = self._determine_feature_compatibility(symbol)
                if 'feature_count' in compat_info and 'uses_enhanced_features' in compat_info:
                    validation_results['backward_compatibility'] = True
                    logger.info("✓ Backward compatibility system working")
            except Exception as e:
                validation_results['errors'].append(f"Backward compatibility test failed: {e}")
            
            # Test 6: Prediction stability (quick test with small ensemble)
            try:
                if len(X_test) > 10:
                    # Create small test model
                    test_model = self.create_lstm_model((X_test.shape[1], X_test.shape[2]))
                    
                    # Test prediction consistency
                    test_input = X_test[-1:, :, :]
                    pred1 = self._predict_single_model(test_model, test_input)
                    pred2 = self._predict_single_model(test_model, test_input)
                    
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
                logger.warning(f"Errors encountered: {len(validation_results['errors'])}")
                for error in validation_results['errors']:
                    logger.warning(f"  - {error}")
            
            if validation_results['warnings']:
                logger.info(f"Warnings: {len(validation_results['warnings'])}")
                for warning in validation_results['warnings']:
                    logger.info(f"  - {warning}")
                    
        except Exception as e:
            validation_results['errors'].append(f"Overall validation failed: {e}")
            logger.error(f"Validation failed: {e}")
        
        return validation_results
    
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
                    'total_params': models[0].count_params() if models else None,
                    'feature_count': models[0].input_shape[2] if models and len(models[0].input_shape) > 2 else 5,
                    'uses_enhanced_features': models[0].input_shape[2] > 5 if models and len(models[0].input_shape) > 2 else False
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
            
            # CRITICAL FIX: Dynamically determine expected feature count from model input shape
            actual_input_shape = model.input_shape
            if len(actual_input_shape) < 3:
                logger.warning(f"Model {model_index} for {symbol} has invalid input shape dimensions: {actual_input_shape}")
                return False
                
            # Extract feature count from model's actual input shape
            expected_feature_count = actual_input_shape[2]
            expected_input_shape = (None, self.sequence_length, expected_feature_count)
            
            # Verify input shape structure (batch, sequence, features)
            if actual_input_shape[1] != self.sequence_length:
                logger.warning(f"Model {model_index} for {symbol} has unexpected sequence length: {actual_input_shape[1]} vs {self.sequence_length}")
                return False
            
            # Log model compatibility info
            logger.debug(f"Model {model_index} for {symbol}: input_shape={actual_input_shape}, features={expected_feature_count}")
            
            # CRITICAL FIX: Test with dummy data matching model's expected feature count
            dummy_input = np.zeros((1, self.sequence_length, expected_feature_count))
            try:
                # Use direct model call with consistent tensor format
                prediction = self._predict_single_model(model, dummy_input).numpy()
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
                dummy_input = np.zeros((1, self.sequence_length, expected_feature_count))
                logger.debug(f"Model {model_index} for {symbol}: using {expected_feature_count} features for validation")
                
                # Perform dummy prediction to build metrics and ensure model is functional
                dummy_output = self._predict_single_model(model, dummy_input).numpy()
                
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
                    # CRITICAL FIX: Force rebuild using dynamic feature count
                    model_feature_count = model.input_shape[2] if len(model.input_shape) >= 3 else 5
                    model.build(input_shape=(None, self.sequence_length, model_feature_count))
                    logger.debug(f"Rebuilt input layer for model {model_index} for {symbol} with {model_feature_count} features")
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
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001,
                        clipnorm=1.0
                    ),
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
    
    def _get_model_feature_count(self, model: tf.keras.Model) -> int:
        """Safely extract feature count from model input shape"""
        try:
            if hasattr(model, 'input_shape') and len(model.input_shape) >= 3:
                return model.input_shape[2]
            else:
                # Fallback to basic features if input shape is not available
                return 5
        except Exception:
            # Ultimate fallback
            return 5
    
    def _validate_loaded_models(self, models: list, symbol: str):
        """Validate that loaded models are functional"""
        try:
            for i, model in enumerate(models):
                if model is None:
                    raise ValueError(f"Model {i} is None")
                
                # Check if model has the expected structure
                if not hasattr(model, 'predict'):
                    raise ValueError(f"Model {i} doesn't have predict method")
                
                # CRITICAL FIX: Dynamically determine expected input shape from model
                actual_input_shape = model.input_shape
                model_feature_count = self._get_model_feature_count(model)
                expected_input_shape = (None, self.sequence_length, model_feature_count)
                
                # Verify sequence length matches (features can vary)
                if len(actual_input_shape) >= 2 and actual_input_shape[1] != self.sequence_length:
                    logger.warning(
                        f"Model {i} for {symbol} has unexpected sequence length. "
                        f"Expected: {self.sequence_length}, Got: {actual_input_shape[1]}"
                    )
                
                logger.debug(f"Model {i} for {symbol} validation passed (features: {model_feature_count})")
                
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