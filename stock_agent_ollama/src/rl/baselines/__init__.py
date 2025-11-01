"""Baseline trading strategies for comparison."""

from .buy_hold import BuyHoldStrategy, buy_hold_strategy
from .momentum import MomentumStrategy, SimpleMomentumStrategy, momentum_strategy

__all__ = [
    'BuyHoldStrategy',
    'buy_hold_strategy',
    'MomentumStrategy',
    'SimpleMomentumStrategy',
    'momentum_strategy',
]
