"""
Buy-and-hold baseline strategy.
"""

import numpy as np
from typing import Optional


class BuyHoldStrategy:
    """
    Simple buy-and-hold strategy.

    Buys on the first step and holds until the end.
    """

    def __init__(self):
        """Initialize buy-hold strategy."""
        self.has_bought = False

    def get_action(self, observation: np.ndarray, **kwargs) -> int:
        """
        Get action for buy-hold strategy.

        Args:
            observation: Current observation (unused)
            **kwargs: Additional parameters

        Returns:
            Action (0: SELL, 1: HOLD, 2: BUY_SMALL, 3: BUY_LARGE)
        """
        if not self.has_bought:
            self.has_bought = True
            return 3  # BUY_LARGE on first step
        else:
            return 1  # HOLD for all subsequent steps

    def reset(self):
        """Reset strategy state."""
        self.has_bought = False


def buy_hold_strategy(observation: np.ndarray, state: Optional[dict] = None) -> int:
    """
    Stateless buy-hold strategy function.

    Args:
        observation: Current observation
        state: Optional state dictionary

    Returns:
        Action
    """
    if state is None:
        state = {'step': 0}

    if state['step'] == 0:
        state['step'] += 1
        return 3  # BUY_LARGE
    else:
        state['step'] += 1
        return 1  # HOLD
