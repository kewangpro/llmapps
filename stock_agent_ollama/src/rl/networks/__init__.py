"""
Neural network architectures for RL agents.
"""

from .lstm_extractor import (
    LSTMFeatureExtractor,
    LSTMExtractorWithAttention,
    create_lstm_feature_extractor
)

__all__ = [
    'LSTMFeatureExtractor',
    'LSTMExtractorWithAttention',
    'create_lstm_feature_extractor'
]
