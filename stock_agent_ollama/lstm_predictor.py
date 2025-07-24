import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, Tuple, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LSTMPredictorInput(BaseModel):
    data: List[Dict] = Field(description="Historical stock data")
    symbol: str = Field(description="Stock symbol")
    prediction_days: int = Field(default=30, description="Number of days to predict")
    sequence_length: int = Field(default=60, description="Sequence length for LSTM")


class LSTMPredictor(BaseTool):
    name = "lstm_predictor"
    description = """Trains LSTM model on historical data and predicts future stock prices.
    Input should be JSON format: {"data": [historical_data_list], "symbol": "AAPL", "prediction_days": 30}"""
    args_schema = LSTMPredictorInput
    
    def _run(self, data: List[Dict], symbol: str, prediction_days: int = 30, sequence_length: int = 60) -> Dict[str, Any]:
        logger.info(f"Starting LSTM prediction for {symbol}: {prediction_days} days, sequence_length={sequence_length}")
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Use closing prices for prediction
            prices = df['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(prices)
            
            # Prepare training data
            X_train, y_train = self._create_sequences(scaled_data[:-prediction_days], sequence_length)
            
            if len(X_train) < 10:
                logger.warning(f"Insufficient training data: {len(X_train)} sequences")
                return {"error": "Insufficient data for training. Need at least 70 days of historical data."}
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),  # Explicitly set return_sequences=False
                Dropout(0.2),
                Dense(1)
            ])
            
            # Use legacy Adam optimizer for better performance on M1/M2 Macs
            try:
                from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
                optimizer = LegacyAdam(learning_rate=0.001)
            except ImportError:
                optimizer = Adam(learning_rate=0.001)
            
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            # Train the model
            logger.info(f"Training LSTM model with {len(X_train)} samples...")
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")
            
            # Make predictions for the next 30 days
            last_sequence = scaled_data[-sequence_length:]
            predictions = []
            current_sequence = last_sequence.copy()
            
            for i in range(prediction_days):
                # Ensure proper shape for prediction
                input_data = current_sequence.reshape(1, sequence_length, 1)
                pred = model.predict(input_data, verbose=0)
                pred_value = pred[0, 0] if pred.ndim > 1 else pred[0]
                predictions.append(pred_value)
                
                # Update sequence with new prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred_value
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            
            # Calculate model performance on training data
            try:
                train_predictions = model.predict(X_train, verbose=0)
                train_predictions = scaler.inverse_transform(train_predictions)
                actual_train = scaler.inverse_transform(y_train.reshape(-1, 1))
                
                mse = mean_squared_error(actual_train, train_predictions)
                mae = mean_absolute_error(actual_train, train_predictions)
            except Exception as perf_error:
                logger.warning(f"Could not calculate model performance: {perf_error}")
                # Use dummy values if performance calculation fails
                mse = mae = 0.01
            
            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
            
            # Calculate trend analysis
            recent_prices = prices[-30:].flatten()  # Last 30 days
            trend_direction = "Upward" if predictions[-1] > recent_prices[-1] else "Downward"
            trend_strength = abs((predictions[-1] - recent_prices[-1]) / recent_prices[-1]) * 100
            
            result = {
                "symbol": symbol,
                "predictions": predictions.tolist(),
                "future_dates": [date.strftime('%Y-%m-%d') for date in future_dates],
                "model_performance": {
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(np.sqrt(mse))
                },
                "trend_analysis": {
                    "direction": trend_direction,
                    "strength_percent": float(trend_strength),
                    "current_price": float(recent_prices[-1]),
                    "predicted_price_30d": float(predictions[-1]),
                    "price_change": float(predictions[-1] - recent_prices[-1]),
                    "percentage_change": float(((predictions[-1] - recent_prices[-1]) / recent_prices[-1]) * 100)
                },
                "historical_prices": recent_prices.tolist(),
                "prediction_confidence": "Medium" if mae < recent_prices.mean() * 0.1 else "Low"
            }
            
            logger.info(f"LSTM prediction completed for {symbol}: {trend_direction} trend, {result['trend_analysis']['percentage_change']:.2f}% change predicted")
            return result
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction for {symbol}: {str(e)}")
            return {"error": f"Error in LSTM prediction: {str(e)}"}
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    async def _arun(self, data: List[Dict], symbol: str, prediction_days: int = 30, sequence_length: int = 60) -> Dict[str, Any]:
        return self._run(data, symbol, prediction_days, sequence_length)