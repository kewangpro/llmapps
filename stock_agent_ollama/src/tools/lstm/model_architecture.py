import tensorflow as tf
from typing import Tuple

def create_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
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
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First LSTM layer with batch normalization
    lstm1 = tf.keras.layers.LSTM(lstm_units[0], return_sequences=True, 
                               recurrent_dropout=0.1)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(lstm1)
    drop1 = tf.keras.layers.Dropout(dropout_rate)(bn1)
    
    # Second LSTM layer
    lstm2 = tf.keras.layers.LSTM(lstm_units[1], return_sequences=True,
                               recurrent_dropout=0.1)(drop1)
    bn2 = tf.keras.layers.BatchNormalization()(lstm2)
    drop2 = tf.keras.layers.Dropout(dropout_rate)(bn2)
    
    # Attention Mechanism
    attention_output = tf.keras.layers.Attention()([drop2, drop2]) # Query and Value from same source
    attention_dense = tf.keras.layers.Dense(lstm_units[1], activation='relu')(attention_output)
    
    # Third LSTM layer (now takes attention output)
    lstm3 = tf.keras.layers.LSTM(lstm_units[2], return_sequences=False,
                               recurrent_dropout=0.1)(attention_dense) # Feed attention output here
    bn3 = tf.keras.layers.BatchNormalization()(lstm3)
    drop3 = tf.keras.layers.Dropout(dropout_rate)(bn3)
    
    # Dense layers with regularization
    dense1 = tf.keras.layers.Dense(dense_units, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(drop3)
    drop4 = tf.keras.layers.Dropout(0.2)(dense1)
    outputs = tf.keras.layers.Dense(1)(drop4)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
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
