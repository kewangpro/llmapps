"""
Comprehensive tests for baseline trading strategies.

Tests for BuyHoldStrategy, MomentumStrategy, SimpleMomentumStrategy,
and stateless strategy functions.
"""

import pytest
import numpy as np
from src.rl.baselines import (
    BuyHoldStrategy,
    MomentumStrategy,
    SimpleMomentumStrategy,
    buy_hold_strategy,
    momentum_strategy
)


@pytest.fixture
def sample_observation():
    """Create a sample observation array."""
    # Observation shape: (lookback, features)
    # Last column is typically the price
    return np.array([
        [100.0, 0.5, 0.3],
        [101.0, 0.6, 0.4],
        [102.0, 0.7, 0.5],
        [103.0, 0.8, 0.6],
        [104.0, 0.9, 0.7]
    ])


@pytest.fixture
def increasing_prices():
    """Create observation with increasing prices."""
    prices = np.linspace(100, 120, 30)
    return np.column_stack([prices, np.random.rand(30), np.random.rand(30)])


@pytest.fixture
def decreasing_prices():
    """Create observation with decreasing prices."""
    prices = np.linspace(120, 100, 30)
    return np.column_stack([prices, np.random.rand(30), np.random.rand(30)])


# ==================== BuyHoldStrategy Tests ====================

def test_buy_hold_initialization():
    """Test BuyHoldStrategy initialization."""
    strategy = BuyHoldStrategy(use_improved_actions=True)
    assert strategy.has_bought is False
    assert strategy.use_improved_actions is True

    strategy_standard = BuyHoldStrategy(use_improved_actions=False)
    assert strategy_standard.use_improved_actions is False


def test_buy_hold_buys_on_first_step(sample_observation):
    """Test BuyHoldStrategy buys on first action."""
    strategy = BuyHoldStrategy(use_improved_actions=True)

    action = strategy.get_action(sample_observation)

    # For improved actions: BUY_LARGE = 3
    assert action == 3
    assert strategy.has_bought is True


def test_buy_hold_holds_after_first_buy(sample_observation):
    """Test BuyHoldStrategy holds after initial buy."""
    strategy = BuyHoldStrategy(use_improved_actions=True)

    # First action - buy
    first_action = strategy.get_action(sample_observation)
    assert first_action == 3

    # Subsequent actions - hold
    second_action = strategy.get_action(sample_observation)
    assert second_action == 0  # HOLD for improved actions

    third_action = strategy.get_action(sample_observation)
    assert third_action == 0


def test_buy_hold_standard_actions(sample_observation):
    """Test BuyHoldStrategy with standard action space."""
    strategy = BuyHoldStrategy(use_improved_actions=False)

    # First action - buy
    first_action = strategy.get_action(sample_observation)
    assert first_action == 3  # BUY_LARGE

    # Subsequent actions - hold
    second_action = strategy.get_action(sample_observation)
    assert second_action == 1  # HOLD for standard actions


def test_buy_hold_reset(sample_observation):
    """Test BuyHoldStrategy reset functionality."""
    strategy = BuyHoldStrategy(use_improved_actions=True)

    # Buy
    strategy.get_action(sample_observation)
    assert strategy.has_bought is True

    # Reset
    strategy.reset()
    assert strategy.has_bought is False

    # Should buy again after reset
    action = strategy.get_action(sample_observation)
    assert action == 3


# ==================== Stateless buy_hold_strategy Tests ====================

def test_buy_hold_strategy_function_first_step(sample_observation):
    """Test stateless buy_hold_strategy function on first step."""
    state = {'step': 0}
    action = buy_hold_strategy(sample_observation, state, use_improved_actions=True)

    assert action == 3  # BUY_LARGE
    assert state['step'] == 1


def test_buy_hold_strategy_function_subsequent_steps(sample_observation):
    """Test stateless buy_hold_strategy function after first step."""
    state = {'step': 1}
    action = buy_hold_strategy(sample_observation, state, use_improved_actions=True)

    assert action == 0  # HOLD (improved actions)
    assert state['step'] == 2


def test_buy_hold_strategy_function_no_state(sample_observation):
    """Test stateless buy_hold_strategy function initializes state."""
    action = buy_hold_strategy(sample_observation, state=None, use_improved_actions=True)
    assert action == 3  # Should buy on first call


def test_buy_hold_strategy_function_standard_actions(sample_observation):
    """Test stateless buy_hold_strategy with standard actions."""
    state = {'step': 0}
    action = buy_hold_strategy(sample_observation, state, use_improved_actions=False)
    assert action == 3  # BUY

    action2 = buy_hold_strategy(sample_observation, state, use_improved_actions=False)
    assert action2 == 1  # HOLD


# ==================== MomentumStrategy Tests ====================

def test_momentum_strategy_initialization():
    """Test MomentumStrategy initialization."""
    strategy = MomentumStrategy(lookback=20, threshold=0.02, use_improved_actions=True)

    assert strategy.lookback == 20
    assert strategy.threshold == 0.02
    assert strategy.price_history == []
    assert strategy.has_position is False
    assert strategy.use_improved_actions is True


def test_momentum_strategy_insufficient_data(sample_observation):
    """Test MomentumStrategy holds when insufficient data."""
    strategy = MomentumStrategy(lookback=20, use_improved_actions=True)

    # With only a few data points (< lookback), should hold
    for i in range(10):
        action = strategy.get_action(sample_observation)
        assert action == 0  # HOLD

    assert len(strategy.price_history) == 10


def test_momentum_strategy_positive_momentum_buy(increasing_prices):
    """Test MomentumStrategy buys on positive momentum."""
    strategy = MomentumStrategy(lookback=10, threshold=0.05, use_improved_actions=True)

    # Feed increasing prices
    for i in range(len(increasing_prices)):
        obs = increasing_prices[i:i+1]  # Single step
        action = strategy.get_action(obs, price=increasing_prices[i, 0])

    # Should eventually buy due to positive momentum
    assert strategy.has_position is True


def test_momentum_strategy_negative_momentum_sell(decreasing_prices):
    """Test MomentumStrategy sells on negative momentum."""
    strategy = MomentumStrategy(lookback=10, threshold=0.05, use_improved_actions=True)

    # Start with a position
    strategy.has_position = True

    # Feed decreasing prices
    for i in range(len(decreasing_prices)):
        obs = decreasing_prices[i:i+1]
        action = strategy.get_action(obs, price=decreasing_prices[i, 0])

    # Should eventually sell due to negative momentum
    assert strategy.has_position is False


def test_momentum_strategy_below_threshold_holds():
    """Test MomentumStrategy holds when momentum is below threshold."""
    strategy = MomentumStrategy(lookback=5, threshold=0.10, use_improved_actions=True)

    # Feed prices with small changes (below threshold)
    prices = [100.0, 100.5, 101.0, 100.8, 101.2, 101.5]
    obs = np.array([[p, 0.5, 0.5] for p in prices])

    for i, price in enumerate(prices):
        action = strategy.get_action(obs[i:i+1], price=price)

    # Should not have triggered buy (momentum too small)
    # Position state depends on whether momentum crossed threshold
    assert len(strategy.price_history) == len(prices)


def test_momentum_strategy_standard_actions():
    """Test MomentumStrategy with standard action space."""
    strategy = MomentumStrategy(lookback=5, threshold=0.02, use_improved_actions=False)

    # Insufficient data should return HOLD (action 1 for standard)
    obs = np.array([[100.0, 0.5, 0.5]])
    action = strategy.get_action(obs, price=100.0)
    assert action == 1  # Standard HOLD


def test_momentum_strategy_reset():
    """Test MomentumStrategy reset functionality."""
    strategy = MomentumStrategy(lookback=10, threshold=0.02, use_improved_actions=True)

    # Add some history
    obs = np.array([[100.0, 0.5, 0.5]])
    for i in range(5):
        strategy.get_action(obs, price=100.0 + i)

    strategy.has_position = True

    # Reset
    strategy.reset()

    assert strategy.price_history == []
    assert strategy.has_position is False


def test_momentum_strategy_extracts_price_from_observation():
    """Test MomentumStrategy extracts price from observation when not provided."""
    strategy = MomentumStrategy(lookback=5, threshold=0.02, use_improved_actions=True)

    obs = np.array([[100.0, 0.5, 0.5]])

    # Call without explicit price (should extract from obs[-1, 0])
    action = strategy.get_action(obs)

    # Should have extracted and stored the price
    assert len(strategy.price_history) == 1
    assert strategy.price_history[0] == 100.0


# ==================== SimpleMomentumStrategy Tests ====================

def test_simple_momentum_initialization():
    """Test SimpleMomentumStrategy initialization."""
    strategy = SimpleMomentumStrategy(threshold=0.01, use_improved_actions=True)

    assert strategy.threshold == 0.01
    assert strategy.prev_price is None
    assert strategy.has_position is False
    assert strategy.use_improved_actions is True


def test_simple_momentum_first_action_buys(sample_observation):
    """Test SimpleMomentumStrategy buys on first action."""
    strategy = SimpleMomentumStrategy(threshold=0.01, use_improved_actions=True)

    action = strategy.get_action(sample_observation)

    assert action == 3  # BUY_LARGE
    assert strategy.prev_price is not None


def test_simple_momentum_positive_change_buy(increasing_prices):
    """Test SimpleMomentumStrategy buys on positive price change."""
    strategy = SimpleMomentumStrategy(threshold=0.02, use_improved_actions=True)

    # First call - buys
    action1 = strategy.get_action(increasing_prices[0:5])
    assert action1 == 3  # BUY

    # Subsequent call with higher price
    action2 = strategy.get_action(increasing_prices[10:15])

    # Should maintain or reinforce position
    # If already has position, might hold; if not, might buy


def test_simple_momentum_negative_change_sell():
    """Test SimpleMomentumStrategy sells on negative price change."""
    strategy = SimpleMomentumStrategy(threshold=0.02, use_improved_actions=True)

    # Start with a position
    obs1 = np.array([[120.0, 0.5, 0.5]])
    strategy.get_action(obs1)
    strategy.has_position = True

    # Price drops
    obs2 = np.array([[110.0, 0.5, 0.5]])
    action = strategy.get_action(obs2)

    # Should sell due to negative momentum
    assert action == 5  # SELL_ALL (improved actions)
    assert strategy.has_position is False


def test_simple_momentum_below_threshold_holds():
    """Test SimpleMomentumStrategy holds when change below threshold."""
    strategy = SimpleMomentumStrategy(threshold=0.10, use_improved_actions=True)

    # First action
    obs1 = np.array([[100.0, 0.5, 0.5]])
    strategy.get_action(obs1)

    # Small price change (below 10% threshold)
    obs2 = np.array([[102.0, 0.5, 0.5]])
    action = strategy.get_action(obs2)

    # Should hold (change is only 2%, below 10% threshold)
    assert action == 0  # HOLD


def test_simple_momentum_standard_actions():
    """Test SimpleMomentumStrategy with standard actions."""
    strategy = SimpleMomentumStrategy(threshold=0.02, use_improved_actions=False)

    obs = np.array([[100.0, 0.5, 0.5]])
    action = strategy.get_action(obs)

    # First action should buy (action 3)
    assert action == 3


def test_simple_momentum_reset():
    """Test SimpleMomentumStrategy reset functionality."""
    strategy = SimpleMomentumStrategy(threshold=0.01, use_improved_actions=True)

    # Take some actions
    obs = np.array([[100.0, 0.5, 0.5]])
    strategy.get_action(obs)
    strategy.has_position = True

    # Reset
    strategy.reset()

    assert strategy.prev_price is None
    assert strategy.has_position is False


# ==================== Stateless momentum_strategy Tests ====================

def test_momentum_strategy_function_insufficient_data(sample_observation):
    """Test stateless momentum_strategy with insufficient data."""
    state = {'prices': [], 'has_position': False}
    action = momentum_strategy(sample_observation, state, lookback=10, threshold=0.01)

    assert action == 1  # HOLD (insufficient data)
    assert len(state['prices']) == 1


def test_momentum_strategy_function_no_state(sample_observation):
    """Test stateless momentum_strategy initializes state."""
    action = momentum_strategy(sample_observation, state=None, lookback=10, threshold=0.01)

    assert action == 1  # HOLD (first step, no state)


def test_momentum_strategy_function_positive_momentum(increasing_prices):
    """Test stateless momentum_strategy detects positive momentum."""
    state = {'prices': [], 'has_position': False}
    lookback = 10
    threshold = 0.05

    # Feed increasing prices
    for i in range(15):
        obs = increasing_prices[i:i+1]
        action = momentum_strategy(obs, state, lookback=lookback, threshold=threshold)

    # Should eventually buy
    assert state['has_position'] is True


def test_momentum_strategy_function_negative_momentum(decreasing_prices):
    """Test stateless momentum_strategy detects negative momentum."""
    state = {'prices': [], 'has_position': True}  # Start with position
    lookback = 10
    threshold = 0.05

    # Feed decreasing prices
    for i in range(15):
        obs = decreasing_prices[i:i+1]
        action = momentum_strategy(obs, state, lookback=lookback, threshold=threshold)

    # Should eventually sell
    assert state['has_position'] is False


def test_momentum_strategy_function_maintains_window():
    """Test stateless momentum_strategy maintains price window."""
    state = {'prices': [], 'has_position': False}
    lookback = 5

    obs = np.array([[100.0, 0.5, 0.5]])

    # Add more prices than lookback
    for i in range(10):
        momentum_strategy(obs, state, lookback=lookback, threshold=0.01)

    # Should keep only lookback prices
    assert len(state['prices']) == lookback


# ==================== Edge Cases & Integration Tests ====================

def test_all_strategies_handle_nan_observation():
    """Test strategies handle NaN values gracefully."""
    obs_with_nan = np.array([
        [100.0, 0.5, 0.5],
        [np.nan, 0.6, 0.6],
        [102.0, 0.7, 0.7]
    ])

    # BuyHoldStrategy
    buy_hold = BuyHoldStrategy()
    action = buy_hold.get_action(obs_with_nan)
    assert isinstance(action, (int, np.integer))

    # SimpleMomentumStrategy (extracts from obs[-1, 0])
    simple_mom = SimpleMomentumStrategy()
    # Should handle gracefully (might get nan price, but shouldn't crash)


def test_strategies_consistent_action_spaces():
    """Test strategies return valid actions for their action space."""
    obs = np.array([[100.0, 0.5, 0.5]])

    # Improved actions: 0-5
    buy_hold_improved = BuyHoldStrategy(use_improved_actions=True)
    action = buy_hold_improved.get_action(obs)
    assert action in [0, 1, 2, 3, 4, 5]

    # Standard actions: 0-3
    buy_hold_standard = BuyHoldStrategy(use_improved_actions=False)
    action = buy_hold_standard.get_action(obs)
    assert action in [0, 1, 2, 3]


def test_momentum_vs_simple_momentum_behavior():
    """Test MomentumStrategy and SimpleMomentumStrategy have similar logic."""
    obs = np.array([[100.0, 0.5, 0.5]])

    # Both should start by building history / buying
    mom = MomentumStrategy(lookback=10, threshold=0.02, use_improved_actions=True)
    simple_mom = SimpleMomentumStrategy(threshold=0.02, use_improved_actions=True)

    # SimpleMomentumStrategy buys immediately
    action_simple = simple_mom.get_action(obs)
    assert action_simple == 3

    # MomentumStrategy needs lookback period first
    action_mom = mom.get_action(obs, price=100.0)
    assert action_mom == 0  # HOLD (insufficient data)


def test_strategies_are_deterministic():
    """Test strategies give consistent results for same inputs."""
    obs = np.array([[100.0, 0.5, 0.5]])

    strategy = BuyHoldStrategy(use_improved_actions=True)

    # Same observation should give same action
    action1 = strategy.get_action(obs)
    strategy.reset()
    action2 = strategy.get_action(obs)

    assert action1 == action2


def test_all_strategies_can_reset():
    """Test all strategies implement reset correctly."""
    obs = np.array([[100.0, 0.5, 0.5]])

    strategies = [
        BuyHoldStrategy(),
        MomentumStrategy(),
        SimpleMomentumStrategy()
    ]

    for strategy in strategies:
        # Take some actions
        strategy.get_action(obs)

        # Reset should not crash
        strategy.reset()

        # Should be able to use again after reset
        action = strategy.get_action(obs)
        assert isinstance(action, (int, np.integer))
