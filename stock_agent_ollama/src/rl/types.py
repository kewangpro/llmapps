from enum import IntEnum

class TradingAction(IntEnum):
    """Discrete trading actions."""
    SELL = 0
    HOLD = 1
    BUY_SMALL = 2
    BUY_LARGE = 3

class ImprovedTradingAction(IntEnum):
    """Improved discrete trading actions with HOLD as default (action 0)."""
    HOLD = 0          # Default action - do nothing
    BUY_SMALL = 1     # Buy with 10-20% of cash (adaptive)
    BUY_MEDIUM = 2    # Buy with 20-40% of cash (adaptive)
    BUY_LARGE = 3     # Buy with 40-60% of cash (adaptive)
    SELL_PARTIAL = 4  # Sell 50% of position
    SELL_ALL = 5      # Sell entire position
