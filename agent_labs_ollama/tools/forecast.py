#!/usr/bin/env python3
"""
Forecast Tool
Uses GRU or N-BEATS neural networks to predict future data points based on time series data

GRU (Gated Recurrent Unit) is the default model - uses multi-horizon forecasting to predict
all future steps simultaneously, preserving trends and patterns better than iterative methods.

N-BEATS (Neural Basis Expansion Analysis) is available as an experimental alternative
for datasets with strong patterns.
"""

import json
import sys
import os
import base64
import io
import signal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Lambda, Concatenate, Add
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import warnings

# Suppress warnings and optimize for production
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure TensorFlow for production
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Enable memory growth to prevent OOM
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Forecast operation timed out")

def save_to_outputs_folder(content: str, filename: str) -> str:
    """Save content to outputs folder and return the full path"""
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path

def create_gru_model(input_shape: Tuple[int, int], forecast_horizon: int = 1) -> Sequential:
    """
    Create GRU model architecture - faster and often more accurate than LSTM

    Args:
        input_shape: (time_steps, features) shape for input sequences
        forecast_horizon: Number of future steps to predict (1 for single-step, >1 for multi-horizon)
    """
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.15),
        GRU(64, return_sequences=True),
        Dropout(0.15),
        GRU(32),
        Dropout(0.15),
        Dense(16, activation='relu'),
        Dense(forecast_horizon)  # Output layer predicts 'forecast_horizon' steps
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def create_nbeats_block(input_layer, block_id: int, units: int, lookback: int, forecast_horizon: int):
    """
    Create a single N-BEATS block (generic architecture)
    Each block learns a basis expansion for backcast and forecast
    """
    # Fully connected stack
    fc = Dense(units, activation='relu', name=f'block_{block_id}_fc1')(input_layer)
    fc = Dense(units, activation='relu', name=f'block_{block_id}_fc2')(fc)
    fc = Dense(units, activation='relu', name=f'block_{block_id}_fc3')(fc)
    fc = Dense(units, activation='relu', name=f'block_{block_id}_fc4')(fc)

    # Backcast (reconstruction of input)
    backcast = Dense(lookback, activation='linear', name=f'block_{block_id}_backcast')(fc)

    # Forecast (prediction of future)
    forecast = Dense(forecast_horizon, activation='linear', name=f'block_{block_id}_forecast')(fc)

    return backcast, forecast

def create_nbeats_model(lookback: int, forecast_horizon: int, num_blocks: int = 2, units: int = 64) -> Model:
    """
    Create N-BEATS (Neural Basis Expansion Analysis for Time Series) model

    N-BEATS uses a stack of blocks, each producing a backcast (residual) and forecast.
    The architecture ensures smooth transitions and interpretable decomposition.

    Args:
        lookback: Number of historical time steps
        forecast_horizon: Number of future time steps to predict
        num_blocks: Number of stacked blocks (default: 2 for faster training)
        units: Hidden units per block (default: 64 for smaller model)
    """
    input_layer = Input(shape=(lookback,), name='input')

    # Store residuals and forecasts from each block
    residuals = input_layer
    forecast_outputs = []

    for block_id in range(num_blocks):
        # Create block
        backcast, forecast = create_nbeats_block(
            residuals, block_id, units, lookback, forecast_horizon
        )

        # Subtract backcast from residuals (doubly residual architecture)
        residuals = Lambda(lambda x: x[0] - x[1], name=f'residual_{block_id}')([residuals, backcast])

        # Store forecast
        forecast_outputs.append(forecast)

    # Sum all forecasts (hierarchical aggregation)
    if len(forecast_outputs) > 1:
        final_forecast = Add(name='forecast_sum')(forecast_outputs)
    else:
        final_forecast = forecast_outputs[0]

    model = Model(inputs=input_layer, outputs=final_forecast, name='nbeats')

    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for stability
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def calculate_trend_momentum(data: np.array, window: int = 20) -> float:
    """
    Calculate recent trend momentum from historical data
    Returns the average rate of change over the last 'window' periods
    """
    if len(data) < window:
        window = len(data)

    recent_data = data[-window:].flatten()

    # Calculate linear regression slope for trend direction
    x = np.arange(len(recent_data))
    coeffs = np.polyfit(x, recent_data, 1)
    momentum = coeffs[0]  # Slope represents trend momentum

    return momentum

def prepare_sequence_data(data: np.array, time_steps: int = 60, forecast_horizon: int = 1) -> Tuple[np.array, np.array]:
    """
    Prepare data for GRU/RNN training - supports both univariate and multivariate

    Args:
        data: Input time series data, shape (samples, features) or (samples, 1)
        time_steps: Lookback window size
        forecast_horizon: Number of steps to predict ahead (1 for single-step, >1 for multi-horizon)

    Returns:
        X: Input sequences of shape (samples, time_steps, features) for multivariate
           or (samples, time_steps) for univariate
        y: Target sequences of shape (samples,) for single-step or (samples, forecast_horizon) for multi-horizon
           Note: y always uses only the first feature (target variable)
    """
    X, y = [], []
    num_features = data.shape[1] if len(data.shape) > 1 else 1

    for i in range(time_steps, len(data) - forecast_horizon + 1):
        if num_features > 1:
            # Multivariate: X includes all features
            X.append(data[i-time_steps:i, :])  # Shape: (time_steps, num_features)
        else:
            # Univariate: X includes single feature
            X.append(data[i-time_steps:i, 0])  # Shape: (time_steps,)

        if forecast_horizon == 1:
            # Single-step: predict next value of target (first feature)
            y.append(data[i, 0])
        else:
            # Multi-horizon: predict next 'forecast_horizon' steps of target
            y.append(data[i:i+forecast_horizon, 0])

    return np.array(X), np.array(y)

def prepare_nbeats_data(data: np.array, lookback: int, forecast_horizon: int) -> Tuple[np.array, np.array]:
    """
    Prepare data for N-BEATS training

    Args:
        data: Scaled time series data
        lookback: Number of historical time steps (input window)
        forecast_horizon: Number of future time steps (output window)

    Returns:
        X: Input sequences of shape (samples, lookback)
        y: Target sequences of shape (samples, forecast_horizon)
    """
    X, y = [], []

    for i in range(lookback, len(data) - forecast_horizon + 1):
        # Input: lookback window
        X.append(data[i-lookback:i, 0])
        # Output: forecast_horizon window
        y.append(data[i:i+forecast_horizon, 0])

    return np.array(X), np.array(y)

def create_forecast_visualization(historical_data: pd.DataFrame, predictions: np.array,
                                date_column: str, value_column: str, forecast_periods: int) -> str:
    """Create forecast visualization and return as base64"""
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(historical_data[date_column], historical_data[value_column],
             label='Historical Data', color='blue', linewidth=2)

    # Create future dates for predictions
    last_date = pd.to_datetime(historical_data[date_column].iloc[-1])
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods)

    # Plot predictions
    plt.plot(future_dates, predictions,
             label='Forecast', color='red', linewidth=2, linestyle='--')

    plt.title('Time Series Forecast', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64

def detect_time_granularity(dates: pd.Series) -> str:
    """Detect time granularity (daily, weekly, monthly, etc.)"""
    if len(dates) < 2:
        return "daily"

    # Calculate median time difference
    date_diffs = pd.to_datetime(dates).diff().dropna()
    median_diff = date_diffs.median()

    if median_diff <= pd.Timedelta(days=1):
        return "daily"
    elif median_diff <= pd.Timedelta(days=7):
        return "weekly"
    elif median_diff <= pd.Timedelta(days=31):
        return "monthly"
    elif median_diff <= pd.Timedelta(days=92):
        return "quarterly"
    else:
        return "yearly"

def generate_future_dates(last_date: pd.Timestamp, periods: int, granularity: str) -> pd.DatetimeIndex:
    """Generate future dates based on detected granularity"""
    if granularity == "monthly":
        # Use month offset for monthly data
        return pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
    elif granularity == "weekly":
        return pd.date_range(start=last_date, periods=periods + 1, freq='W')[1:]
    elif granularity == "quarterly":
        return pd.date_range(start=last_date, periods=periods + 1, freq='QS')[1:]
    elif granularity == "yearly":
        return pd.date_range(start=last_date, periods=periods + 1, freq='YS')[1:]
    else:  # daily
        return pd.date_range(start=last_date + timedelta(days=1), periods=periods)

def _forecast_single_series_data(df: pd.DataFrame, date_column: str, value_column: str,
                                 forecast_periods: int, time_steps: int, model_type: str,
                                 min_data_points: int = 30) -> tuple:
    """Forecast a single time series and return predictions with model"""
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    # Prepare data
    values = df[value_column].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    # Check if we have enough data (reduced minimum for multi-series)
    if len(scaled_data) < min_data_points:
        raise ValueError(f"Insufficient data for forecasting. Need at least {min_data_points} data points, got {len(scaled_data)}")

    # Adjust time_steps based on available data (more aggressive for small datasets)
    if len(scaled_data) < 20:
        # For very small datasets (8-19 points), use minimal lookback
        time_steps = max(3, len(scaled_data) // 3)
    elif len(scaled_data) < time_steps + 20:
        time_steps = max(10, min(40, len(scaled_data) // 2))

    # Prepare training data
    X, y = prepare_sequence_data(scaled_data, time_steps, forecast_horizon=forecast_periods)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Create and train model
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model = create_gru_model((time_steps, 1), forecast_horizon=forecast_periods)
    model.fit(X_train, y_train, epochs=100, batch_size=32,
             validation_data=(X_val, y_val), verbose=0,
             shuffle=False, callbacks=[early_stop])

    # Make predictions
    last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    predictions = model.predict(last_sequence, verbose=0).flatten()

    # Scale back
    predictions = predictions.reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()

    # Apply bounds
    historical_mean = values.mean()
    historical_std = values.std()
    min_bound = max(0, historical_mean - 4 * historical_std)
    max_bound = historical_mean + 4 * historical_std
    predictions = np.clip(predictions, min_bound, max_bound)

    return predictions, model, X_val, y_val, scaler

def _forecast_multi_series(df: pd.DataFrame, date_column: str, value_column: str,
                           group_column: str, forecast_periods: int, time_steps: int,
                           model_type: str) -> Dict[str, Any]:
    """Forecast multiple time series (one per group)"""
    try:
        print(f"Starting multi-series forecast for {df[group_column].nunique()} groups", file=sys.stderr)

        # Detect time granularity from first group
        first_group = df[df[group_column] == df[group_column].iloc[0]]
        granularity = detect_time_granularity(first_group[date_column])
        print(f"Detected time granularity: {granularity}", file=sys.stderr)

        all_forecasts = []
        all_historical = []
        overall_metrics = []

        groups = df[group_column].unique()

        # Determine minimum data points based on first group size
        first_group_size = len(first_group)
        min_data_points = max(6, min(8, first_group_size))  # At least 6, prefer 8

        # Adjust forecast_periods for small datasets (don't forecast more than we have data)
        adjusted_forecast_periods = forecast_periods
        if first_group_size < 20:
            # For very small datasets, need to ensure: time_steps + forecast_periods <= data_points
            # Estimate time_steps will be about data_points // 3
            estimated_time_steps = max(3, first_group_size // 3)
            max_forecast = first_group_size - estimated_time_steps - 1
            # Limit forecast to half of historical data for safety
            adjusted_forecast_periods = min(forecast_periods, max(4, first_group_size // 2))
            print(f"Adjusted forecast_periods from {forecast_periods} to {adjusted_forecast_periods} for small dataset ({first_group_size} points)", file=sys.stderr)

        print(f"Using minimum data points: {min_data_points} (based on {first_group_size} points per group)", file=sys.stderr)

        for group_name in groups:
            try:
                group_df = df[df[group_column] == group_name].copy()

                # Skip groups with insufficient data (less than minimum)
                if len(group_df) < min_data_points:
                    print(f"Skipping {group_name}: insufficient data ({len(group_df)} points, need {min_data_points})", file=sys.stderr)
                    continue

                # Forecast this group with reduced minimum requirement
                predictions, model, X_val, y_val, scaler = _forecast_single_series_data(
                    group_df, date_column, value_column, adjusted_forecast_periods, time_steps, model_type,
                    min_data_points=min_data_points  # Pass reduced minimum
                )

                # Generate future dates based on granularity (use adjusted periods)
                last_date = pd.to_datetime(group_df[date_column].max())
                future_dates = generate_future_dates(last_date, adjusted_forecast_periods, granularity)

                # Store forecasts (CSV output - predictions only, no series_type column)
                for date, value in zip(future_dates, predictions):
                    all_forecasts.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "business_unit": group_name,
                        "predicted_value": float(value)
                    })

                # Store historical data
                for idx, row in group_df.iterrows():
                    all_historical.append({
                        "date": pd.to_datetime(row[date_column]).strftime("%Y-%m-%d"),
                        "business_unit": group_name,
                        "value": float(row[value_column]),
                        "series_type": "Historical"
                    })

                # Calculate metrics
                val_predictions = model.predict(X_val, verbose=0).flatten().reshape(-1, 1)
                y_val_flat = y_val.flatten().reshape(-1, 1)
                val_predictions_scaled = scaler.inverse_transform(val_predictions)
                y_val_scaled = scaler.inverse_transform(y_val_flat)

                mae = mean_absolute_error(y_val_scaled, val_predictions_scaled)
                overall_metrics.append(mae)

                print(f"Completed forecast for {group_name}: {len(predictions)} predictions", file=sys.stderr)

            except Exception as e:
                print(f"Error forecasting {group_name}: {str(e)}", file=sys.stderr)
                continue

        if not all_forecasts:
            return {
                "success": False,
                "error": "No forecasts could be generated for any group",
                "forecast_data": [],
                "visualization": None
            }

        # Save ONLY forecast data to CSV file (not historical)
        forecast_only_df = pd.DataFrame(all_forecasts)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forecast_filename = f"forecast_{timestamp}.csv"
        forecast_csv = forecast_only_df.to_csv(index=False)
        output_path = save_to_outputs_folder(forecast_csv, forecast_filename)

        # Create downloadable file data (forecast only)
        file_base64 = base64.b64encode(forecast_csv.encode('utf-8')).decode('utf-8')
        file_size_mb = len(forecast_csv.encode('utf-8')) / (1024 * 1024)

        # Format forecast data for backward compatibility
        forecast_data_output = []
        for item in all_forecasts:
            forecast_data_output.append({
                "date": item["date"],
                "predicted_value": item["predicted_value"],
                "business_unit": item["business_unit"]
            })

        return {
            "success": True,
            "forecast_data": forecast_data_output,
            "visualization": None,  # Multi-series visualization is complex, let viz tool handle it
            "output_path": output_path,
            "model_metrics": {
                "mae": float(np.mean(overall_metrics)) if overall_metrics else 0,
                "mse": 0.0,  # Not calculated for multi-series (would need to aggregate across groups)
                "rmse": 0.0,  # Not calculated for multi-series (would need to aggregate across groups)
                "groups_forecasted": len(all_forecasts) // adjusted_forecast_periods if adjusted_forecast_periods > 0 else 0
            },
            "data_info": {
                "historical_points": len(all_historical),
                "forecast_points": len(all_forecasts),
                "groups": len(groups),
                "forecast_periods": adjusted_forecast_periods,  # Use actual periods used
                "forecast_periods_requested": forecast_periods,  # Original request
                "time_granularity": granularity,
                "date_column": date_column,
                "value_column": value_column,
                "group_column": group_column
            },
            "file_size_mb": round(file_size_mb, 4),
            "forecast_file_data": {
                "base64": file_base64,
                "filename": forecast_filename,
                "mime_type": "text/csv",
                "content_preview": forecast_csv[:300] + "..." if len(forecast_csv) > 300 else forecast_csv
            }
            # NOTE: No tool_data for multi-series - orchestrator will combine cost_analysis + forecast
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Multi-series forecast error: {str(e)}",
            "forecast_data": [],
            "visualization": None
        }

def forecast_timeseries(data: str, forecast_periods: int = 30, time_steps: int = 60,
                       date_column: str = None, value_column: str = None,
                       group_column: str = None, model_type: str = "gru",
                       feature_columns: str = None) -> Dict[str, Any]:
    """
    Forecast future values using GRU or N-BEATS neural network
    Supports multi-series forecasting when group_column is provided
    Supports multivariate forecasting when feature_columns is provided

    Args:
        data: CSV data containing time series
        forecast_periods: Number of periods to forecast
        time_steps: Number of time steps for lookback
        date_column: Name of date column
        value_column: Name of value column (target variable to forecast)
        group_column: Name of grouping column (e.g., 'business_unit') for separate forecasts
        model_type: Model to use - "gru" (default, more stable) or "nbeats" (experimental)
        feature_columns: Comma-separated additional feature columns for multivariate forecasting
                        (e.g., "open,high,low,volume,rsi,volatility,ma_20,ma_50")

    Returns:
        Dictionary with forecast results and visualization
    """
    # Set timeout for production deployment (5 minutes)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes timeout

    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(data))

        if df.empty:
            return {
                "success": False,
                "error": "No data found in input",
                "forecast_data": [],
                "visualization": None
            }

        # Auto-detect columns if not provided
        if date_column is None:
            # Look for common date column names
            date_candidates = ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp', 'Timestamp', 'month', 'Month']
            for col in date_candidates:
                if col in df.columns:
                    date_column = col
                    break
            if date_column is None:
                date_column = df.columns[0]

        if value_column is None:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_column = numeric_cols[0]
            else:
                value_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # Auto-detect group column if not provided (for multi-series forecasting)
        if group_column is None and len(df.columns) > 2:
            # Look for categorical columns that could represent different series
            for col in df.columns:
                # Exclude "series" column (used in stock data for Price/MA indicators, not for grouping)
                if col not in [date_column, value_column, 'series'] and df[col].dtype == 'object':
                    # Check if this looks like a grouping column
                    unique_count = df[col].nunique()
                    if 2 <= unique_count <= 100:  # Reasonable number of groups
                        group_column = col
                        print(f"Auto-detected grouping column: {group_column} ({unique_count} groups)", file=sys.stderr)
                        break

        # Ensure we have the required columns
        if date_column not in df.columns or value_column not in df.columns:
            return {
                "success": False,
                "error": f"Required columns not found. Available columns: {list(df.columns)}",
                "forecast_data": [],
                "visualization": None
            }

        # Check if we have multi-series data
        if group_column and group_column in df.columns:
            print(f"Multi-series forecasting enabled for column: {group_column}", file=sys.stderr)
            return _forecast_multi_series(df, date_column, value_column, group_column,
                                         forecast_periods, time_steps, model_type)
        else:
            print("Single series forecasting", file=sys.stderr)
            # Continue with single series forecasting (existing logic)

        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=date_column)

        # Prepare data - MULTIVARIATE or UNIVARIATE
        is_multivariate = feature_columns is not None and feature_columns.strip() != ""

        if is_multivariate:
            # Multivariate forecasting: target + additional features
            feature_list = [value_column] + [f.strip() for f in feature_columns.split(',')]

            # Filter out any features that don't exist in DataFrame
            available_features = [f for f in feature_list if f in df.columns]
            missing_features = [f for f in feature_list if f not in df.columns]

            if missing_features:
                print(f"Warning: Missing features in data: {missing_features}", file=sys.stderr)

            if len(available_features) < 2:
                # Not enough features, fall back to univariate
                print(f"Warning: Not enough features available, falling back to univariate", file=sys.stderr)
                is_multivariate = False
                values = df[value_column].values.reshape(-1, 1)
            else:
                # Extract all feature columns
                values = df[available_features].values  # Shape: (samples, num_features)
                num_features = values.shape[1]
                print(f"Multivariate forecasting with {num_features} features: {available_features}", file=sys.stderr)
        else:
            # Univariate forecasting: only target variable
            values = df[value_column].values.reshape(-1, 1)
            num_features = 1

        # Scale the data (each feature independently)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(values)

        # Check if we have enough data - reduce requirements for production
        if len(scaled_data) < 30:
            return {
                "success": False,
                "error": f"Insufficient data for forecasting. Need at least 30 data points, got {len(scaled_data)}",
                "forecast_data": [],
                "visualization": None
            }

        # Adjust time_steps based on available data (keep reasonable lookback)
        if len(scaled_data) < time_steps + 20:
            time_steps = max(20, min(40, len(scaled_data) // 2))

        # Import early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        if model_type == "nbeats":
            # N-BEATS: Direct multi-step forecasting
            # Prepare data for N-BEATS
            X, y = prepare_nbeats_data(scaled_data, time_steps, forecast_periods)

            if len(X) < 10:
                # Fall back to GRU if not enough data for N-BEATS
                model_type = "gru"
            else:
                # Split data (use last 20% for validation)
                split_index = int(len(X) * 0.8)
                X_train, X_val = X[:split_index], X[split_index:]
                y_train, y_val = y[:split_index], y[split_index:]

                # Create and train N-BEATS model with regularization
                model = create_nbeats_model(time_steps, forecast_periods, num_blocks=2, units=32)

                history = model.fit(
                    X_train, y_train,
                    epochs=30,  # Fewer epochs to prevent overfitting
                    batch_size=32,  # Larger batch for smoother gradients
                    validation_data=(X_val, y_val),
                    verbose=0,
                    shuffle=False,
                    callbacks=[early_stop]
                )

                # Make prediction with N-BEATS (single multi-step prediction)
                last_sequence = scaled_data[-time_steps:, 0].reshape(1, -1)
                predictions = model.predict(last_sequence, verbose=0)
                predictions = predictions.flatten()

        if model_type == "gru":
            # GRU: Direct multi-horizon forecasting (predicts all steps at once)
            # Prepare training data for multi-horizon forecasting
            X, y = prepare_sequence_data(scaled_data, time_steps, forecast_horizon=forecast_periods)

            if is_multivariate:
                # Multivariate: X already has shape (samples, time_steps, num_features)
                # y has shape (samples, forecast_periods) - only target variable
                pass  # No reshaping needed, prepare_sequence_data handles it
            else:
                # Univariate: X has shape (samples, time_steps), reshape to (samples, time_steps, 1)
                X = X.reshape(X.shape[0], X.shape[1], 1)

            # Split data (use last 20% for validation)
            split_index = int(len(X) * 0.8)
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]

            # Create and train model with multi-horizon output
            # Input shape: (time_steps, num_features)
            model = create_gru_model((time_steps, num_features), forecast_horizon=forecast_periods)

            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0,
                shuffle=False,
                callbacks=[early_stop]
            )

            # Make predictions using GRU model (direct multi-horizon)
            if is_multivariate:
                last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, num_features)
            else:
                last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)

            # Predict all future steps at once
            predictions = model.predict(last_sequence, verbose=0)
            predictions = predictions.flatten()

        # Scale predictions back to original scale
        predictions = predictions.reshape(-1, 1)

        if is_multivariate:
            # For multivariate, create a dedicated scaler for JUST the target column
            # The predictions are only for the target (first column), not all features
            target_values = df[value_column].values.reshape(-1, 1)
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaler.fit(target_values)

            print(f"[DEBUG] Multivariate inverse transform:", file=sys.stderr)
            print(f"  Target column: {value_column}", file=sys.stderr)
            print(f"  Target min: {target_values.min():.2f}, max: {target_values.max():.2f}", file=sys.stderr)
            print(f"  Predictions (scaled) min: {predictions.min():.4f}, max: {predictions.max():.4f}", file=sys.stderr)
            print(f"  Last 3 historical values: {target_values[-3:].flatten()}", file=sys.stderr)

            # Use target-specific scaler for inverse transform
            predictions = target_scaler.inverse_transform(predictions)

            print(f"  Predictions (unscaled) min: {predictions.min():.2f}, max: {predictions.max():.2f}", file=sys.stderr)
            print(f"  First 3 predictions: {predictions[:3].flatten()}", file=sys.stderr)
        else:
            # For univariate, use the same scaler directly
            predictions = scaler.inverse_transform(predictions)

        predictions = predictions.flatten()

        # Apply reasonable bounds based on RECENT historical trend, not overall mean
        # This prevents forecasts from reverting to historical mean when there's a strong trend
        recent_window = min(30, len(values) // 4)  # Last 30 days or 25% of data
        recent_values = values[-recent_window:, 0] if is_multivariate else values[-recent_window:, 0]
        recent_mean = recent_values.mean()
        recent_std = recent_values.std()

        # Use last known value as anchor point
        last_value = values[-1, 0] if is_multivariate else values[-1, 0]

        # Allow predictions to deviate ±50% from last value, or ±4σ from recent mean, whichever is wider
        deviation_from_last = abs(last_value * 0.5)
        deviation_from_trend = 4 * recent_std
        max_deviation = max(deviation_from_last, deviation_from_trend)

        min_bound = max(0, last_value - max_deviation)
        max_bound = last_value + max_deviation

        # Soft clipping - allow deviation while preventing extreme outliers
        predictions = np.clip(predictions, min_bound, max_bound)

        # Create future dates
        last_date = df[date_column].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods)

        # Prepare forecast data
        forecast_data = []
        for i, (date, value) in enumerate(zip(future_dates, predictions)):
            forecast_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "predicted_value": float(value),
                "period": i + 1
            })

        # Create visualization
        visualization = create_forecast_visualization(df, predictions, date_column, value_column, forecast_periods)

        # Calculate model metrics on validation set
        val_predictions = model.predict(X_val, verbose=0)

        if model_type == "nbeats" or model_type == "gru":
            # Both N-BEATS and GRU now output multi-step predictions
            # Flatten predictions and targets for metrics calculation
            val_predictions_flat = val_predictions.flatten().reshape(-1, 1)
            y_val_flat = y_val.flatten().reshape(-1, 1)

            if is_multivariate:
                # Use target-specific scaler (same as predictions above)
                val_predictions_scaled = target_scaler.inverse_transform(val_predictions_flat)
                y_val_scaled = target_scaler.inverse_transform(y_val_flat)
            else:
                # Use full scaler for univariate
                val_predictions_scaled = scaler.inverse_transform(val_predictions_flat)
                y_val_scaled = scaler.inverse_transform(y_val_flat)
        else:
            # Fallback for other model types
            if is_multivariate:
                val_predictions_scaled = target_scaler.inverse_transform(val_predictions)
                y_val_scaled = target_scaler.inverse_transform(y_val.reshape(-1, 1))
            else:
                val_predictions_scaled = scaler.inverse_transform(val_predictions)
                y_val_scaled = scaler.inverse_transform(y_val.reshape(-1, 1))

        mae = mean_absolute_error(y_val_scaled, val_predictions_scaled)
        mse = mean_squared_error(y_val_scaled, val_predictions_scaled)
        rmse = np.sqrt(mse)

        # Save forecast to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forecast_filename = f"forecast_{timestamp}.csv"
        forecast_df = pd.DataFrame(forecast_data)
        forecast_csv = forecast_df.to_csv(index=False)
        output_path = save_to_outputs_folder(forecast_csv, forecast_filename)

        # Create downloadable file data
        file_base64 = base64.b64encode(forecast_csv.encode('utf-8')).decode('utf-8')
        file_size_mb = len(forecast_csv.encode('utf-8')) / (1024 * 1024)

        return {
            "success": True,
            "forecast_data": forecast_data,
            "visualization": visualization,
            "output_path": output_path,
            "model_metrics": {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse)
            },
            "data_info": {
                "historical_points": len(df),
                "forecast_periods": forecast_periods,
                "time_steps": time_steps,
                "date_column": date_column,
                "value_column": value_column
            },
            "file_size_mb": round(file_size_mb, 4),
            "forecast_file_data": {
                "base64": file_base64,
                "filename": forecast_filename,
                "mime_type": "text/csv",
                "content_preview": forecast_csv[:300] + "..." if len(forecast_csv) > 300 else forecast_csv
            }
        }

    except TimeoutException as e:
        return {
            "success": False,
            "error": "Forecast operation timed out. Please try with smaller dataset or fewer forecast periods.",
            "forecast_data": [],
            "visualization": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "forecast_data": [],
            "visualization": None
        }
    finally:
        # Clear the alarm
        signal.alarm(0)

def main():
    if len(sys.argv) < 2:
        result = {
            "success": False,
            "error": "Usage: python forecast.py '{\"data\": \"csv_data\", \"forecast_periods\": 30}'"
        }
        print(json.dumps(result))
        return

    try:
        params = json.loads(sys.argv[1])
        data = params.get("data", "")
        forecast_periods = params.get("forecast_periods", 30)
        time_steps = params.get("time_steps", 60)
        date_column = params.get("date_column")
        value_column = params.get("value_column")
        group_column = params.get("group_column")  # Support for grouping column
        model_type = params.get("model_type", "gru")
        feature_columns = params.get("feature_columns")  # NEW: Support for multivariate features

        result = forecast_timeseries(data, forecast_periods, time_steps, date_column, value_column,
                                    group_column, model_type, feature_columns)
        print(json.dumps(result))

    except json.JSONDecodeError as e:
        result = {
            "success": False,
            "error": f"Invalid JSON parameters: {str(e)}"
        }
        print(json.dumps(result))
    except Exception as e:
        result = {
            "success": False,
            "error": f"Forecast error: {str(e)}"
        }
        print(json.dumps(result))

if __name__ == "__main__":
    main()