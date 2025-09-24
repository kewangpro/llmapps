#!/usr/bin/env python3
"""
Forecast Tool
Uses LSTM neural networks to predict future data points based on time series data
"""

import json
import sys
import os
import base64
import io
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def save_to_outputs_folder(content: str, filename: str) -> str:
    """Save content to outputs folder and return the full path"""
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path

def create_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def prepare_lstm_data(data: np.array, time_steps: int = 60) -> Tuple[np.array, np.array]:
    """Prepare data for LSTM training"""
    X, y = [], []

    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])

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

    plt.title('Time Series Forecast using LSTM', fontsize=16, fontweight='bold')
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
                       date_column: str = None, value_column: str = None) -> Dict[str, Any]:
    """
    Forecast future values using LSTM neural network

    Args:
        data: CSV data containing time series
        forecast_periods: Number of periods to forecast
        time_steps: Number of time steps for LSTM lookback
        date_column: Name of date column
        value_column: Name of value column

    Returns:
        Dictionary with forecast results and visualization
    """
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

        # Check if we have enough data
        if len(scaled_data) < time_steps + 20:
            time_steps = max(10, len(scaled_data) // 3)
            if len(scaled_data) < time_steps + 10:
                return {
                    "success": False,
                    "error": f"Insufficient data for forecasting. Need at least {time_steps + 10} data points, got {len(scaled_data)}",
                    "forecast_data": [],
                    "visualization": None
                }

        # Prepare training data
        X, y = prepare_lstm_data(scaled_data, time_steps)

        # Reshape for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split data (use last 20% for validation)
        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        # Create and train model
        model = create_lstm_model((time_steps, 1))

        # Train with early stopping
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0,
            shuffle=False
        )

        # Make predictions
        last_sequence = scaled_data[-time_steps:]
        predictions = []

        for _ in range(forecast_periods):
            # Reshape for prediction
            current_sequence = last_sequence.reshape(1, time_steps, 1)

            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])

            # Update sequence for next prediction
            last_sequence = np.append(last_sequence[1:], next_pred[0, 0])

        # Scale predictions back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        predictions = predictions.flatten()

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
        val_predictions = scaler.inverse_transform(val_predictions)
        y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1))

        mae = mean_absolute_error(y_val_original, val_predictions)
        mse = mean_squared_error(y_val_original, val_predictions)
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

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "forecast_data": [],
            "visualization": None
        }

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

        result = forecast_timeseries(data, forecast_periods, time_steps, date_column, value_column)
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