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
    Prepare data for GRU/RNN training

    Args:
        data: Input time series data
        time_steps: Lookback window size
        forecast_horizon: Number of steps to predict ahead (1 for single-step, >1 for multi-horizon)

    Returns:
        X: Input sequences of shape (samples, time_steps)
        y: Target sequences of shape (samples,) for single-step or (samples, forecast_horizon) for multi-horizon
    """
    X, y = [], []

    for i in range(time_steps, len(data) - forecast_horizon + 1):
        X.append(data[i-time_steps:i, 0])
        if forecast_horizon == 1:
            y.append(data[i, 0])
        else:
            # Multi-horizon: predict next 'forecast_horizon' steps
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

def forecast_timeseries(data: str, forecast_periods: int = 30, time_steps: int = 60,
                       date_column: str = None, value_column: str = None, model_type: str = "gru") -> Dict[str, Any]:
    """
    Forecast future values using GRU or N-BEATS neural network

    Args:
        data: CSV data containing time series
        forecast_periods: Number of periods to forecast
        time_steps: Number of time steps for lookback
        date_column: Name of date column
        value_column: Name of value column
        model_type: Model to use - "gru" (default, more stable) or "nbeats" (experimental)

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
            date_candidates = ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp', 'Timestamp']
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

        # Ensure we have the required columns
        if date_column not in df.columns or value_column not in df.columns:
            return {
                "success": False,
                "error": f"Required columns not found. Available columns: {list(df.columns)}",
                "forecast_data": [],
                "visualization": None
            }

        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=date_column)

        # Prepare data
        values = df[value_column].values.reshape(-1, 1)

        # Scale the data
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

            # Reshape for GRU
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Split data (use last 20% for validation)
            split_index = int(len(X) * 0.8)
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]

            # Create and train model with multi-horizon output
            model = create_gru_model((time_steps, 1), forecast_horizon=forecast_periods)

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
            last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)

            # Predict all future steps at once
            predictions = model.predict(last_sequence, verbose=0)
            predictions = predictions.flatten()

        # Scale predictions back to original scale
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        predictions = predictions.flatten()

        # Apply reasonable bounds based on historical data statistics
        # Use wider bounds (±4σ) to allow more variation while preventing extremes
        historical_mean = values.mean()
        historical_std = values.std()
        min_bound = max(0, historical_mean - 4 * historical_std)
        max_bound = historical_mean + 4 * historical_std

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

            val_predictions_scaled = scaler.inverse_transform(val_predictions_flat)
            y_val_scaled = scaler.inverse_transform(y_val_flat)
        else:
            # Fallback for other model types
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
        model_type = params.get("model_type", "gru")

        result = forecast_timeseries(data, forecast_periods, time_steps, date_column, value_column, model_type)
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