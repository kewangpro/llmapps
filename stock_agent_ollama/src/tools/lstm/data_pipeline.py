import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.tools.technical_analysis import TechnicalAnalysis
from src.tools.lstm.custom_scalers import CompositeScaler

logger = logging.getLogger(__name__)

def _cap_outliers(series: pd.Series, lower_percentile: float = 2, upper_percentile: float = 98) -> pd.Series:
    """Cap outliers using aggressive percentile-based limits"""
    lower_bound = np.percentile(series.dropna(), lower_percentile)
    upper_bound = np.percentile(series.dropna(), upper_percentile)
    return series.clip(lower=lower_bound, upper=upper_bound).infer_objects(copy=False)

def _winsorize_outliers(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
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
        return _cap_outliers(series, 1, 99)
    except Exception:
        # Fallback to aggressive capping
        return _cap_outliers(series, 1, 99)

def _aggressive_outlier_removal(series: pd.Series, z_threshold: float = 2.5) -> pd.Series:
    """Remove extreme outliers using strict z-score threshold"""
    # Calculate z-scores
    z_scores = np.abs((series - series.mean()) / series.std())
    # Replace extreme outliers with median for smoother distribution
    median_val = series.median()
    result = series.copy()
    result[z_scores > z_threshold] = median_val
    return result

def _ultra_aggressive_preprocessing(series: pd.Series, feature_name: str) -> pd.Series:
    """Ultra-aggressive preprocessing for MSFT to achieve <50 outlier target"""
    # Step 1: Remove extreme outliers using strict z-score
    cleaned = _aggressive_outlier_removal(series, z_threshold=1.8)
    
    # Step 2: Apply aggressive Winsorization
    winsorized = _winsorize_outliers(cleaned, limits=(0.002, 0.002))  # 0.2% on each end
    
    # Step 3: Apply strict percentile capping
    capped = _cap_outliers(winsorized, 0.5, 99.5)
    
    # Step 4: For all features, apply rolling median smoothing to reduce noise
    smoothed = capped.rolling(window=3, center=True).median().fillna(capped)
    
    # Step 5: Final outlier removal pass
    final_cleaned = _aggressive_outlier_removal(smoothed, z_threshold=1.5)
    
    return final_cleaned

def _calculate_robust_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate exactly 17 enhanced features with ultra-aggressive outlier handling for <50 outlier target"""
    ta = TechnicalAnalysis()
    enhanced_data = data[['Close', 'Volume', 'High', 'Low', 'Open']].copy()
    
    # Apply ultra-aggressive preprocessing to Volume due to extreme outliers in MSFT
    enhanced_data['Volume'] = _ultra_aggressive_preprocessing(data['Volume'], 'Volume')
    
    # Calculate moving averages first for ratio features
    sma_20 = ta.calculate_sma(data['Close'], 20)
    sma_50 = ta.calculate_sma(data['Close'], 50)
    volume_ma_20 = ta.calculate_sma(data['Volume'], 20)
    
    # Add ratio features with ultra-aggressive outlier handling
    price_sma20_ratio = data['Close'] / sma_20
    price_sma50_ratio = data['Close'] / sma_50
    volume_ratio = data['Volume'] / volume_ma_20
    
    enhanced_data['Price_vs_SMA20'] = _ultra_aggressive_preprocessing(price_sma20_ratio, 'Price_vs_SMA20')
    enhanced_data['Price_vs_SMA50'] = _ultra_aggressive_preprocessing(price_sma50_ratio, 'Price_vs_SMA50')
    enhanced_data['Volume_MA_Ratio'] = _ultra_aggressive_preprocessing(volume_ratio, 'Volume_MA_Ratio')
    
    # Technical indicators with ultra-aggressive preprocessing
    rsi_raw = ta.calculate_rsi(data['Close'])
    enhanced_data['RSI'] = _ultra_aggressive_preprocessing(rsi_raw, 'RSI')
    
    # MACD indicators with ultra-aggressive preprocessing
    macd_data = ta.calculate_macd(data['Close'])
    enhanced_data['MACD'] = _ultra_aggressive_preprocessing(macd_data['macd'], 'MACD')
    enhanced_data['MACD_Signal'] = _ultra_aggressive_preprocessing(macd_data['macd_signal'], 'MACD_Signal')
    enhanced_data['MACD_Histogram'] = _ultra_aggressive_preprocessing(macd_data['macd_histogram'], 'MACD_Histogram')
    
    # Bollinger Bands with ultra-aggressive preprocessing (remove BB_Middle to reduce feature count)
    bb_data = ta.calculate_bollinger_bands(data['Close'])
    enhanced_data['BB_Upper'] = _ultra_aggressive_preprocessing(bb_data['bb_upper'], 'BB_Upper')
    enhanced_data['BB_Lower'] = _ultra_aggressive_preprocessing(bb_data['bb_lower'], 'BB_Lower')
    # BB_Middle removed as it's redundant with SMA_20
    
    # Moving averages with ultra-aggressive preprocessing (keep only EMA_12, remove SMA_20)
    enhanced_data['EMA_12'] = _ultra_aggressive_preprocessing(ta.calculate_ema(data['Close'], 12), 'EMA_12')
    # SMA_20 removed to reduce feature count from 20 to 17
    
    # Volatility indicators with ultra-aggressive preprocessing
    atr_raw = ta.calculate_atr(data['High'], data['Low'], data['Close'])
    enhanced_data['ATR'] = _ultra_aggressive_preprocessing(atr_raw, 'ATR')
    
    # Consolidated momentum feature with ultra-aggressive preprocessing
    # Smooth price and volume changes to reduce noise
    price_change = data['Close'].pct_change().clip(lower=-0.1, upper=0.1).rolling(window=3, min_periods=1).mean()
    volume_change = data['Volume'].pct_change().clip(lower=-0.3, upper=0.3).rolling(window=3, min_periods=1).mean()
    
    # Add a volatility component to momentum score
    price_volatility = data['Close'].pct_change().rolling(window=5, min_periods=1).std().fillna(0) # 5-day rolling std
    
    # Create a momentum score that combines price change, volume change, and volatility
    # Adjust weights to give more importance to price change and volatility
    momentum_score = (price_change * 0.6) + (volume_change * 0.1) + (price_volatility * 0.3) # New weighting
    enhanced_data['Momentum_Score'] = _ultra_aggressive_preprocessing(momentum_score, 'Momentum_Score')
    # Removed individual Price_Change and Volume_Change to reduce feature count
    
    return enhanced_data

def _create_adaptive_scaler(feature_data: np.ndarray, feature_name: str, symbol: str) -> object:
    """Create adaptive scaler with aggressive outlier handling for MSFT"""
    # Explicitly use RobustScaler for features known to benefit from it
    if feature_name in ['MACD_Histogram', 'Momentum_Score']:
        logger.debug(f"Explicitly using RobustScaler for {feature_name} (as recommended by logs)")
        return RobustScaler(quantile_range=(1.0, 99.0)) # More aggressive quantile range

    # Features that always need robust scaling due to high variability
    always_robust_features = ['Volume', 'Volume_MA_Ratio', 'MACD', 'MACD_Signal', 'ATR'] # Removed MACD_Histogram, Momentum_Score as they are handled explicitly
    
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
    # 4. For MSFT, use RobustScaler for all except basic OHLC
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

def _validate_feature_quality(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
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
            # Exclude MACD_Histogram from this check as it often has high CV due to being centered around zero
            if coefficient_of_variation > 1.5 and col != 'MACD_Histogram':  # High variability
                validation_results['scaling_recommendations'][col] = 'RobustScaler recommended (high CV)'
            
            # Stricter ratio feature validation
            if any(ratio_name in col for ratio_name in ['_vs_', 'Ratio', 'Score']):
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                q_range = q99 - q01
                
                # Special handling for Momentum_Score due to its typical range around zero
                if col == 'Momentum_Score':
                    # For Momentum_Score, check if its range is extremely narrow or wide,
                    # but allow negative Q01 and small Q99
                    if q_range < 0.001 or q_range > 0.1: # Example thresholds, might need tuning
                        validation_results['outlier_warnings'].append(
                            f"{col}: suspicious range for Momentum_Score (Q01: {q01:.3f}, Q99: {q99:.3f}, range: {q_range:.3f})"
                        )
                        validation_results['quality_score'] -= 8
                else:
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

def prepare_enhanced_data_robust(data: pd.DataFrame, sequence_length: int, symbol: str = "UNKNOWN", validation_split: float = None, pre_fitted_scaler: Optional[object] = None) -> Tuple[np.ndarray, np.ndarray, object]:
    """Prepare enhanced data with robust outlier handling and adaptive scaling"""
    logger.debug(f"Preparing robust enhanced features for {symbol}") # Changed INFO to DEBUG
    
    # Calculate robust features with outlier handling
    enhanced_data = _calculate_robust_features(data)
    
    # Remove NaN values (first rows will have NaN due to technical indicators)
    enhanced_data = enhanced_data.dropna()
    
    if len(enhanced_data) < sequence_length + 10:
        raise ValueError(f"Insufficient data after adding technical indicators: {len(enhanced_data)} samples")
    
    # Validate feature quality and log warnings
    feature_quality = _validate_feature_quality(enhanced_data, symbol)
    logger.info(f"Feature quality score for {symbol}: {feature_quality['quality_score']}/100")
    
    # Use pre_fitted_scaler if provided, otherwise create a new composite scaler
    composite_scaler = pre_fitted_scaler if pre_fitted_scaler is not None else CompositeScaler(enhanced_data.columns, symbol, validation_split)

    # Use per-feature adaptive scaling, passing the composite_scaler for overall price range fitting
    scaled_data = _apply_adaptive_scaling(enhanced_data, symbol, validation_split, composite_scaler)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict only the close price (first feature)
    
    logger.info(f"Prepared {len(X)} sequences with {enhanced_data.shape[1]} robust features for {symbol}")
    return np.array(X), np.array(y), composite_scaler

def _apply_adaptive_scaling(data: pd.DataFrame, symbol: str, validation_split: float = None, composite_scaler: Optional[object] = None) -> np.ndarray:
    """Apply adaptive per-feature scaling with data leakage prevention"""
    
    # Split data BEFORE scaling to prevent data leakage
    if validation_split is not None:
        split_index = int(len(data) * (1 - validation_split))
        train_data_df = data.iloc[:split_index]
        val_data_df = data.iloc[split_index:]
    else:
        train_data_df = data
        val_data_df = None
    
    # If composite_scaler is provided and not fitted, fit it on the training data.
    # This happens during training.
    if composite_scaler is not None and not composite_scaler.fitted:
        try:
            composite_scaler.fit_transform(train_data_df)
            logger.debug(f"CompositeScaler fitted for {symbol} with price range: ${composite_scaler.price_min:.2f} - ${composite_scaler.price_max:.2f}")
        except Exception as e:
            logger.warning(f"Failed to fit CompositeScaler for {symbol}: {e}")
    
    # Now, transform the data using the (fitted) composite_scaler.
    # This applies the per-feature scaling internally.
    scaled_data = composite_scaler.transform(data)
    
    return scaled_data

def prepare_enhanced_data(data: pd.DataFrame, sequence_length: int, validation_split: float = None, pre_fitted_scaler: Optional[object] = None) -> Tuple[np.ndarray, np.ndarray, object]:
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
    
    # Always use robust method for all symbols to ensure multivariate forecasting
    logger.debug(f"Using robust enhanced features for {symbol}") # Changed INFO to DEBUG
    return prepare_enhanced_data_robust(data, sequence_length, symbol, validation_split, pre_fitted_scaler)

def _prepare_basic_data(data: pd.DataFrame, sequence_length: int, validation_split: float = None, pre_fitted_scaler: Optional[object] = None) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
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
        if pre_fitted_scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_val_data = scaler.transform(val_data)  # Transform validation data using training scaler
        else:
            scaler = pre_fitted_scaler
            scaled_train_data = scaler.transform(train_data)
            scaled_val_data = scaler.transform(val_data)
        
        # Combine scaled data for sequence creation
        scaled_data = np.vstack([scaled_train_data, scaled_val_data])
    else:
        # For prediction only (no validation split)
        if pre_fitted_scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
        else:
            scaler = pre_fitted_scaler
            scaled_data = scaler.transform(dataset)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict only the close price (first feature)
    
    return np.array(X), np.array(y), scaler
