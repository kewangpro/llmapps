import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from typing import Dict, Any, Tuple, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import warnings
import logging
from data_store import get_data_store
try:
    import talib
except ImportError:
    talib = None
import tensorflow as tf
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TrainingProgressCallback(Callback):
    """Custom callback to track training progress"""
    
    def __init__(self, model_num, total_models, progress_callback=None, symbol=""):
        super().__init__()
        self.model_num = model_num
        self.total_models = total_models
        self.progress_callback = progress_callback
        self.symbol = symbol
        self.epoch_count = 0
        self.total_epochs = 75  # Default from config
        
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get('epochs', 75)
        self.training_stopped_early = False
        logger.info(f"Starting training model {self.model_num}/{self.total_models} for {self.symbol}")
    
    def on_train_end(self, logs=None):
        if self.epoch_count < self.total_epochs:
            self.training_stopped_early = True
            logger.info(f"Model {self.model_num} training ended early at epoch {self.epoch_count}/{self.total_epochs} (likely early stopping)")
        else:
            logger.info(f"Model {self.model_num} completed full training: {self.epoch_count}/{self.total_epochs} epochs")
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count = epoch + 1
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        # Calculate overall progress (across all models and epochs)
        model_progress = (self.model_num - 1) / self.total_models
        epoch_progress = self.epoch_count / self.total_epochs / self.total_models
        total_training_progress = model_progress + epoch_progress
        
        # Update progress if callback provided
        if self.progress_callback:
            # Create detailed progress message
            if val_loss > 0:
                progress_text = f"🧠 Training Model {self.model_num}/{self.total_models} • Epoch {self.epoch_count}/{self.total_epochs} • Loss: {loss:.4f} • Val Loss: {val_loss:.4f}"
            else:
                progress_text = f"🧠 Training Model {self.model_num}/{self.total_models} • Epoch {self.epoch_count}/{self.total_epochs} • Loss: {loss:.4f}"
            
            # Map training progress to step 2 of the 3-step process
            # Step 1: Data fetching (completed)
            # Step 2: LSTM training (in progress) - map 0.0 to 1.0 training progress to step 2.0 to 2.9
            # Step 3: Visualization (pending)
            current_step_float = 2.0 + (total_training_progress * 0.9)  # Maps 0-1 to 2.0-2.9
            current_step = int(current_step_float) + 1 if current_step_float >= 2.9 else 2
            
            try:
                self.progress_callback(current_step, 3, progress_text)
            except Exception as e:
                # Don't let progress callback errors break training
                pass
        
        # Print progress to console
        if self.epoch_count % 10 == 0 or self.epoch_count == self.total_epochs:
            print(f"📈 Model {self.model_num}/{self.total_models} • Epoch {self.epoch_count}/{self.total_epochs} • Loss: {loss:.4f} • Val Loss: {val_loss:.4f}")
        
        logger.info(f"Model {self.model_num}/{self.total_models} - Epoch {self.epoch_count}/{self.total_epochs}: loss={loss:.6f}, val_loss={val_loss:.6f}")


class LSTMPredictorInput(BaseModel):
    symbol: str = Field(description="Stock symbol")
    prediction_days: int = Field(default=30, description="Number of days to predict")
    sequence_length: int = Field(default=120, description="Sequence length for LSTM")
    use_technical_indicators: bool = Field(default=True, description="Include technical indicators")
    ensemble_size: int = Field(default=3, description="Number of models in ensemble")
    progress_callback: Any = Field(default=None, description="Progress update callback function")


class LSTMPredictor(BaseTool):
    name = "lstm_predictor"
    description = """Trains LSTM model on historical data and predicts future stock prices.
    Input should be JSON format: {"symbol": "AAPL", "prediction_days": 30}"""
    args_schema = LSTMPredictorInput
    
    def __init__(self):
        super().__init__()
        # Store progress callback in a way that doesn't conflict with Pydantic
        object.__setattr__(self, '_progress_callback', None)
    
    def set_progress_callback(self, callback):
        """Set the progress callback"""
        object.__setattr__(self, '_progress_callback', callback)
    
    def get_progress_callback(self):
        """Get the progress callback"""
        return getattr(self, '_progress_callback', None)
    
    def _run(self, symbol: str, prediction_days: int = 30, sequence_length: int = 120, 
             use_technical_indicators: bool = True, ensemble_size: int = 3, progress_callback=None) -> Dict[str, Any]:
        
        # Handle case where agent passes JSON string instead of parsed args
        if isinstance(symbol, str) and symbol.startswith('{'):
            logger.warning(f"Received JSON string as symbol: {symbol}")
            import json
            try:
                parsed = json.loads(symbol)
                if 'symbol' in parsed:
                    symbol = parsed['symbol']
                    if 'prediction_days' in parsed:
                        prediction_days = parsed['prediction_days']
                    logger.info(f"Parsed symbol: {symbol}, prediction_days: {prediction_days}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON symbol: {symbol}")
                return {"error": f"Invalid symbol format: {symbol}"}
        
        logger.info(f"Starting LSTM prediction for {symbol}: {prediction_days} days, sequence_length={sequence_length}")
        
        try:
            # Get data from store
            data_store = get_data_store()
            stock_data = data_store.get_stock_data(symbol)
            
            if not stock_data:
                return {"error": f"No historical data found for {symbol}. Please fetch data first using stock_fetcher."}
            
            # Convert data to DataFrame
            data = stock_data['data']
            df = pd.DataFrame(data)
            logger.info(f"DataFrame shape before processing: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            logger.info(f"DataFrame shape after date processing: {df.shape}")
            
            # Add technical indicators if enabled
            if use_technical_indicators:
                features = self._add_technical_indicators(df)
                logger.info(f"Features shape after technical indicators: {features.shape}")
            else:
                features = df[['Close']].values
                logger.info(f"Features shape (Close only): {features.shape}")
            
            # Check for NaN values
            if np.isnan(features).any():
                logger.warning("NaN values detected in features, cleaning...")
                features = np.nan_to_num(features, nan=0.0)
            
            # Use multiple features for prediction
            n_features = features.shape[1]
            logger.info(f"Number of features: {n_features}")
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(features)
            
            # Create separate scaler for target (Close price)
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            target_values = df[['Close']].values
            scaled_target = target_scaler.fit_transform(target_values)
            
            # Prepare training data with validation split
            total_samples = len(scaled_data) - prediction_days
            train_size = int(total_samples * 0.8)
            
            logger.info(f"Total data points: {len(scaled_data)}")
            logger.info(f"Total samples for training: {total_samples}")
            logger.info(f"Train size: {train_size}")
            logger.info(f"Sequence length: {sequence_length}")
            
            # Adjust sequence length if we don't have enough data
            if train_size <= sequence_length:
                new_sequence_length = max(10, train_size // 4)  # Use 1/4 of available data as sequence
                logger.warning(f"Insufficient data for sequence length {sequence_length}. Reducing to {new_sequence_length}")
                sequence_length = new_sequence_length
            
            X_train, y_train = self._create_sequences(scaled_data[:train_size], scaled_target[:train_size], sequence_length)
            X_val, y_val = self._create_sequences(scaled_data[train_size:total_samples], 
                                                scaled_target[train_size:total_samples], sequence_length)
            
            logger.info(f"X_train shape: {X_train.shape if len(X_train) > 0 else 'empty'}")
            logger.info(f"y_train shape: {y_train.shape if len(y_train) > 0 else 'empty'}")
            logger.info(f"X_val shape: {X_val.shape if len(X_val) > 0 else 'empty'}")
            logger.info(f"y_val shape: {y_val.shape if len(y_val) > 0 else 'empty'}")
            
            # Check for any issues with the data
            logger.info(f"X_train contains NaN: {np.isnan(X_train).any()}")
            logger.info(f"y_train contains NaN: {np.isnan(y_train).any()}")
            logger.info(f"X_train dtype: {X_train.dtype}")
            logger.info(f"y_train dtype: {y_train.dtype}")
            
            if len(X_train) < 10:  # Reduced minimum requirement
                logger.warning(f"Insufficient training data: {len(X_train)} sequences")
                return {"error": f"Insufficient data for training. Need at least {sequence_length + 10} days of historical data. Currently have {len(scaled_data)} days."}
            
            # Check validation data
            if len(X_val) == 0:
                logger.warning("No validation data available, using 20% of training data")
                # Split training data for validation
                val_split = int(len(X_train) * 0.8)
                X_val = X_train[val_split:]
                y_val = y_train[val_split:]
                X_train = X_train[:val_split]
                y_train = y_train[:val_split]
                logger.info(f"After split - X_train: {X_train.shape}, X_val: {X_val.shape}")
            
            # Train ensemble of models
            models = []
            predictions_ensemble = []
            
            print(f"\n🔥 Training {ensemble_size} ensemble models on {len(X_train)} sequences...")
            print(f"📊 Features: {n_features}, Sequence length: {sequence_length}")
            print("="*60)
            
            for i in range(ensemble_size):
                model_num = i + 1
                logger.info(f"Training model {model_num}/{ensemble_size}")
                print(f"\n🧠 Starting Model {model_num}/{ensemble_size}")
                
                # Build improved LSTM model with attention
                model = self._build_advanced_model(sequence_length, n_features)
                
                # Compile model
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='huber',  # More robust to outliers than MSE
                    metrics=['mae']
                )
                
                # Setup callbacks including progress tracking
                # Use the tool's progress callback if available, otherwise use the parameter
                active_progress_callback = self.get_progress_callback() or progress_callback
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                    TrainingProgressCallback(model_num, ensemble_size, active_progress_callback, symbol)
                ]
                
                # Train the model
                try:
                    logger.info(f"Starting model.fit() for model {model_num}")
                    logger.info(f"Training data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
                    logger.info(f"Validation data shapes: X_val={X_val.shape}, y_val={y_val.shape}")
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=75,  # Use config value
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=0
                    )
                    logger.info(f"Model {model_num} training completed successfully")
                except Exception as training_error:
                    logger.error(f"Model {model_num} training failed: {str(training_error)}")
                    logger.error(f"Training error type: {type(training_error)}")
                    # Try with a smaller batch size
                    logger.info("Retrying with smaller batch size...")
                    try:
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=75,
                            batch_size=16,  # Smaller batch size
                            callbacks=callbacks,
                            verbose=0
                        )
                        logger.info(f"Model {model_num} training completed with smaller batch size")
                    except Exception as retry_error:
                        logger.error(f"Model {model_num} training failed even with smaller batch: {str(retry_error)}")
                        raise training_error  # Re-raise original error
                
                models.append(model)
                best_val_loss = min(history.history['val_loss'])
                epochs_trained = len(history.history['val_loss'])
                
                if epochs_trained < 75:
                    print(f"✅ Model {model_num} completed with early stopping after {epochs_trained}/75 epochs • Best val_loss: {best_val_loss:.6f}")
                    logger.info(f"Model {model_num} training completed with early stopping after {epochs_trained}/75 epochs. Best val_loss: {best_val_loss:.6f}")
                else:
                    print(f"✅ Model {model_num} completed full training (75/75 epochs) • Best val_loss: {best_val_loss:.6f}")
                    logger.info(f"Model {model_num} training completed full training (75/75 epochs). Best val_loss: {best_val_loss:.6f}")
            
            print(f"\n🎯 All {ensemble_size} models trained successfully!")
            print("="*60)
            
            # Make ensemble predictions
            last_sequence = scaled_data[-sequence_length:]
            
            for model in models:
                model_predictions = []
                current_sequence = last_sequence.copy()
                
                for day in range(prediction_days):
                    input_data = current_sequence.reshape(1, sequence_length, n_features)
                    pred = model.predict(input_data, verbose=0)
                    pred_value = pred[0, 0]
                    model_predictions.append(pred_value)
                    
                    # Update sequence with prediction (only for Close price feature)
                    new_sequence = current_sequence.copy()
                    new_sequence = np.roll(new_sequence, -1, axis=0)
                    new_sequence[-1, 0] = pred_value  # Update Close price
                    # Keep other features from last known values
                    for j in range(1, n_features):
                        new_sequence[-1, j] = current_sequence[-1, j]
                    current_sequence = new_sequence
                
                predictions_ensemble.append(model_predictions)
            
            # Average ensemble predictions
            ensemble_predictions = np.mean(predictions_ensemble, axis=0)
            prediction_std = np.std(predictions_ensemble, axis=0)
            
            # Inverse transform predictions
            predictions = target_scaler.inverse_transform(ensemble_predictions.reshape(-1, 1)).flatten()
            prediction_intervals = target_scaler.inverse_transform(
                (ensemble_predictions + 1.96 * prediction_std).reshape(-1, 1)
            ).flatten() - predictions
            
            # Calculate ensemble model performance on validation data
            val_predictions_ensemble = []
            for model in models:
                val_pred = model.predict(X_val, verbose=0)
                val_predictions_ensemble.append(val_pred.flatten())
            
            val_predictions_mean = np.mean(val_predictions_ensemble, axis=0)
            val_predictions_scaled = target_scaler.inverse_transform(val_predictions_mean.reshape(-1, 1)).flatten()
            val_actual_scaled = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(val_actual_scaled, val_predictions_scaled)
            mae = mean_absolute_error(val_actual_scaled, val_predictions_scaled)
            
            
            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
            
            # Calculate trend analysis
            recent_prices = target_values[-30:].flatten()  # Last 30 days
            trend_direction = "Upward" if predictions[-1] > recent_prices[-1] else "Downward"
            trend_strength = abs((predictions[-1] - recent_prices[-1]) / recent_prices[-1]) * 100
            
            # Calculate prediction confidence based on ensemble std
            confidence_score = 1.0 / (1.0 + np.mean(prediction_std))
            confidence_level = "High" if confidence_score > 0.8 else "Medium" if confidence_score > 0.5 else "Low"
            
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
                "prediction_confidence": confidence_level,
                "prediction_intervals": prediction_intervals.tolist(),
                "ensemble_size": ensemble_size,
                "features_used": n_features
            }
            
            # Store prediction results for visualizer
            data_store.store_stock_data(f"{symbol}_predictions", result)
            
            # Return summary to agent (no large arrays)
            agent_summary = {
                "symbol": symbol,
                "status": "success",
                "message": f"LSTM prediction completed for {symbol}",
                "trend_direction": trend_direction,
                "current_price": float(recent_prices[-1]),
                "predicted_price": float(predictions[-1]),
                "percentage_change": float(((predictions[-1] - recent_prices[-1]) / recent_prices[-1]) * 100),
                "confidence": confidence_level,
                "model_performance": {
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(np.sqrt(mse))
                }
            }
            
            logger.info(f"LSTM prediction completed for {symbol}: {trend_direction} trend, {result['trend_analysis']['percentage_change']:.2f}% change predicted")
            return agent_summary
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction for {symbol}: {str(e)}")
            return {"error": f"Error in LSTM prediction: {str(e)}"}
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i, 0])
        return np.array(X), np.array(y)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> np.ndarray:
        """Add technical indicators to the dataset"""
        features_df = df.copy()
        
        # Price-based features
        features_df['Close'] = df['Close']
        features_df['High'] = df['High']
        features_df['Low'] = df['Low']
        features_df['Volume'] = df['Volume']
        
        # Moving averages
        features_df['SMA_5'] = df['Close'].rolling(window=5).mean()
        features_df['SMA_10'] = df['Close'].rolling(window=10).mean()
        features_df['SMA_20'] = df['Close'].rolling(window=20).mean()
        features_df['EMA_12'] = df['Close'].ewm(span=12).mean()
        features_df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Technical indicators using TA-Lib (with fallbacks)
        use_talib = talib is not None
        if use_talib:
            try:
                features_df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
                macd, macdsignal, macdhist = talib.MACD(df['Close'].values)
                features_df['MACD'] = macd
                features_df['MACD_Signal'] = macdsignal
                features_df['MACD_Hist'] = macdhist
                features_df['BB_Upper'], features_df['BB_Middle'], features_df['BB_Lower'] = talib.BBANDS(df['Close'].values)
                features_df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            except Exception:
                use_talib = False
        
        if not use_talib:
            # Fallback calculations if TA-Lib not available
            # RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['RSI'] = 100 - (100 / (1 + rs))
            
            # Simple MACD
            features_df['MACD'] = features_df['EMA_12'] - features_df['EMA_26']
            features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9).mean()
            features_df['MACD_Hist'] = features_df['MACD'] - features_df['MACD_Signal']
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            features_df['BB_Upper'] = sma_20 + (std_20 * 2)
            features_df['BB_Middle'] = sma_20
            features_df['BB_Lower'] = sma_20 - (std_20 * 2)
            
            # ATR
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            features_df['ATR'] = tr.rolling(window=14).mean()
        
        # Price ratios and volatility
        features_df['Close_SMA20_Ratio'] = df['Close'] / features_df['SMA_20']
        features_df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        features_df['Volume_Ratio'] = df['Volume'] / features_df['Volume_MA']
        features_df['Price_Volatility'] = df['Close'].rolling(window=20).std()
        
        # Select features and handle missing values
        feature_columns = ['Close', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
                          'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                          'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR', 'Close_SMA20_Ratio',
                          'Volume_Ratio', 'Price_Volatility']
        
        features_df = features_df[feature_columns].fillna(method='bfill').fillna(method='ffill').fillna(0)
        return features_df.values
    
    def _build_advanced_model(self, sequence_length: int, n_features: int) -> Model:
        """Build advanced LSTM model with bidirectional layers and attention"""
        # Input layer
        inputs = Input(shape=(sequence_length, n_features))
        
        # First bidirectional LSTM layer
        lstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
        lstm1 = LayerNormalization()(lstm1)
        
        # Second bidirectional LSTM layer
        lstm2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        
        # Simplified attention mechanism
        lstm3 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(lstm2)
        lstm3 = LayerNormalization()(lstm3)
        
        # Dense layers with residual connection
        dense1 = Dense(50, activation='relu', kernel_regularizer=l2(0.001))(lstm3)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(25, activation='relu', kernel_regularizer=l2(0.001))(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        # Output layer
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    async def _arun(self, symbol: str, prediction_days: int = 30, sequence_length: int = 120,
                   use_technical_indicators: bool = True, ensemble_size: int = 3, progress_callback=None) -> Dict[str, Any]:
        return self._run(symbol, prediction_days, sequence_length, use_technical_indicators, ensemble_size, progress_callback)