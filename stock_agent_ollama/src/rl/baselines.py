"""
Baseline trading strategies for comparison.

This module contains simple baseline strategies like Buy & Hold and Momentum.
"""

import numpy as np
from typing import Dict, Any, Optional




# ==================== BUY & HOLD STRATEGY ====================

class BuyHoldStrategy:

    def __init__(self, use_improved_actions: bool = True):
        self.has_bought = False
        self.use_improved_actions = use_improved_actions

    def get_action(self, observation: np.ndarray, **kwargs) -> int:
        """
        Get action for buy-hold strategy.

        Args:
            observation: Current observation (unused)
            **kwargs: Additional parameters

        Returns:
            Action (0: SELL, 1: HOLD, 2: BUY_SMALL, 3: BUY_LARGE for standard)
            Action (0: HOLD, 1: BUY_SMALL, ..., 3: BUY_LARGE for improved)
        """
        if self.use_improved_actions:
            hold_action = 0
            buy_action = 3  # BUY_LARGE
        else:
            hold_action = 1
            buy_action = 3  # BUY_LARGE

        if not self.has_bought:
            self.has_bought = True
            return buy_action
        else:
            return hold_action

    def reset(self):
        """Reset strategy state."""
        self.has_bought = False


def buy_hold_strategy(observation: np.ndarray, state: Optional[dict] = None, use_improved_actions: bool = True) -> int:
    """
    Stateless buy-hold strategy function.
    """
    if state is None:
        state = {'step': 0}

    if use_improved_actions:
        hold_action = 0
        buy_action = 3
    else:
        hold_action = 1
        buy_action = 3

    if state['step'] == 0:
        state['step'] += 1
        return buy_action
    else:
        state['step'] += 1
        return hold_action



# ==================== MOMENTUM STRATEGY ====================

class MomentumStrategy:

    def __init__(self, lookback: int = 20, threshold: float = 0.02, use_improved_actions: bool = True):
        self.lookback = lookback
        self.threshold = threshold
        self.price_history = []
        self.has_position = False
        self.use_improved_actions = use_improved_actions

    def get_action(self, observation: np.ndarray, price: Optional[float] = None, **kwargs) -> int:
        # Extract current price from observation if not provided
        if price is None:
            price = observation[-1, 0]

        self.price_history.append(price)

        if self.use_improved_actions:
            hold_action = 0
            buy_action = 3
            sell_action = 5 # SELL_ALL
        else:
            hold_action = 1
            buy_action = 3
            sell_action = 0

        # Not enough data yet
        if len(self.price_history) < self.lookback:
            return hold_action

        # Calculate momentum
        current_price = self.price_history[-1]
        past_price = self.price_history[-self.lookback]
        momentum = (current_price - past_price) / past_price

        # Trading logic
        if momentum > self.threshold and not self.has_position:
            self.has_position = True
            return buy_action
        elif momentum < -self.threshold and self.has_position:
            self.has_position = False
            return sell_action
        else:
            return hold_action

    def reset(self):
        self.price_history = []
        self.has_position = False


class SimpleMomentumStrategy:
    """
    Simplified momentum strategy using observation directly.
    """

    def __init__(self, threshold: float = 0.0, use_improved_actions: bool = True):
        self.threshold = threshold
        self.prev_price = None
        self.has_position = False
        self.use_improved_actions = use_improved_actions

    def get_action(self, observation: np.ndarray, **kwargs) -> int:
        current_price = observation[-1, 0]

        if self.use_improved_actions:
            hold_action = 0
            buy_action = 3
            sell_action = 5
        else:
            hold_action = 1
            buy_action = 3
            sell_action = 0

        if self.prev_price is None:
            self.prev_price = current_price
            return buy_action

        # Calculate price change
        price_change = (current_price - self.prev_price) / (abs(self.prev_price) + 1e-8)

        # Trading logic
        if price_change > self.threshold and not self.has_position:
            self.has_position = True
            action = buy_action
        elif price_change < -abs(self.threshold) and self.has_position:
            self.has_position = False
            action = sell_action
        else:
            action = hold_action

        self.prev_price = current_price
        return action


    def reset(self):
        """Reset strategy."""
        self.prev_price = None
        self.has_position = False


def momentum_strategy(
    observation: np.ndarray,
    state: Optional[dict] = None,
    lookback: int = 10,
    threshold: float = 0.01
) -> int:
    """
    Stateless momentum strategy function.

    Args:
        observation: Current observation
        state: State dictionary (maintains price history)
        lookback: Lookback period
        threshold: Momentum threshold

    Returns:
        Action
    """
    if state is None:
        state = {'prices': [], 'has_position': False}

    # Get current price from observation
    current_price = observation[-1, 0]
    state['prices'].append(current_price)

    # Keep only recent history
    if len(state['prices']) > lookback:
        state['prices'].pop(0)

    # Not enough data
    if len(state['prices']) < lookback:
        return 1  # HOLD

    # Calculate momentum
    momentum = (state['prices'][-1] - state['prices'][0]) / (abs(state['prices'][0]) + 1e-8)

    # Trading logic
    if momentum > threshold and not state['has_position']:
        state['has_position'] = True
        return 3  # BUY_LARGE
    elif momentum < -threshold and state['has_position']:
        state['has_position'] = False
        return 0  # SELL
    else:
        return 1  # HOLD
