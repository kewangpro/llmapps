"""
LSTM Feature Extractor for temporal pattern extraction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from gymnasium import spaces
import numpy as np


class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based feature extractor for sequential market data.

    Extracts temporal patterns from the observation sequence,
    then produces a fixed-size feature vector for the policy networks.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.0
    ):
        """
        Initialize LSTM feature extractor.

        Args:
            observation_space: Observation space (expects shape: [seq_len, features])
            features_dim: Output feature dimension
            lstm_hidden_size: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
        """
        super().__init__()

        # Extract dimensions from observation space
        # Expected shape: (sequence_length, num_features)
        obs_shape = observation_space.shape
        if len(obs_shape) == 2:
            self.sequence_length, self.input_dim = obs_shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape: {obs_shape}")

        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # Linear projection to desired feature dimension
        self.projection = nn.Linear(lstm_hidden_size, features_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(features_dim)

        # Activation
        self.activation = nn.Tanh()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from sequential observations.

        Args:
            observations: Tensor of shape [batch_size, seq_len, features]

        Returns:
            Extracted features of shape [batch_size, features_dim]
        """
        # observations shape: [batch_size, sequence_length, input_dim]
        batch_size = observations.shape[0]

        # Pass through LSTM
        # lstm_out shape: [batch_size, sequence_length, lstm_hidden_size]
        # hidden shape: [num_layers, batch_size, lstm_hidden_size]
        lstm_out, (hidden, cell) = self.lstm(observations)

        # Use the last hidden state from the last layer
        # hidden[-1] shape: [batch_size, lstm_hidden_size]
        last_hidden = hidden[-1]

        # Project to feature dimension
        features = self.projection(last_hidden)

        # Normalize
        features = self.layer_norm(features)

        # Apply activation
        features = self.activation(features)

        return features


class LSTMExtractorWithAttention(nn.Module):
    """
    Enhanced LSTM feature extractor with attention mechanism.

    Uses attention to focus on important timesteps in the sequence.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.0,
        use_attention: bool = True
    ):
        """
        Initialize LSTM feature extractor with attention.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            lstm_hidden_size: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        # Extract dimensions
        obs_shape = observation_space.shape
        if len(obs_shape) == 2:
            self.sequence_length, self.input_dim = obs_shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape: {obs_shape}")

        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.use_attention = use_attention

        # Bidirectional LSTM for better context
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False  # Keep unidirectional for causal trading
        )

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_hidden_size // 2, 1)
            )

        # Feature projection
        self.projection = nn.Linear(lstm_hidden_size, features_dim)
        self.layer_norm = nn.LayerNorm(features_dim)
        self.activation = nn.Tanh()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features with attention.

        Args:
            observations: Tensor of shape [batch_size, seq_len, features]

        Returns:
            Extracted features of shape [batch_size, features_dim]
        """
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(observations)
        # lstm_out: [batch_size, sequence_length, lstm_hidden_size]

        if self.use_attention:
            # Compute attention weights
            # attention_scores: [batch_size, sequence_length, 1]
            attention_scores = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_scores, dim=1)

            # Weighted sum of LSTM outputs
            # context: [batch_size, lstm_hidden_size]
            context = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            # Use last hidden state
            context = hidden[-1]

        # Project to features
        features = self.projection(context)
        features = self.layer_norm(features)
        features = self.activation(features)

        return features


def create_lstm_feature_extractor(
    observation_space: spaces.Box,
    features_dim: int = 128,
    lstm_hidden_size: int = 64,
    num_lstm_layers: int = 2,
    use_attention: bool = False,
    dropout: float = 0.0
) -> nn.Module:
    """
    Factory function to create LSTM feature extractor.

    Args:
        observation_space: Observation space
        features_dim: Output feature dimension
        lstm_hidden_size: LSTM hidden size
        num_lstm_layers: Number of LSTM layers
        use_attention: Whether to use attention mechanism
        dropout: Dropout rate

    Returns:
        LSTM feature extractor module
    """
    if use_attention:
        return LSTMExtractorWithAttention(
            observation_space=observation_space,
            features_dim=features_dim,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            use_attention=True
        )
    else:
        return LSTMFeatureExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout
        )
