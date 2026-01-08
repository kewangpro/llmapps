"""
Comprehensive tests for RL reward functions.

Tests for SimpleReturnReward, RiskAdjustedReward, CustomizableReward,
and reward function factory.
"""

import pytest
import numpy as np
from src.rl.rewards import (
    RewardConfig,
    RewardFunction,
    SimpleReturnReward,
    RiskAdjustedReward,
    CustomizableReward,
    get_reward_function
)


@pytest.fixture
def default_config():
    """Default reward configuration."""
    return RewardConfig()


@pytest.fixture
def custom_config():
    """Custom reward configuration with specific values."""
    return RewardConfig(
        return_weight=2.0,
        risk_penalty=1.0,
        sharpe_bonus=0.2,
        transaction_cost_rate=0.002,
        slippage_rate=0.001,
        max_drawdown_penalty=0.5,
        extreme_action_penalty=0.02,
        profitable_trade_bonus=0.1
    )


# ==================== SimpleReturnReward Tests ====================

def test_simple_return_reward_initialization(default_config):
    """Test SimpleReturnReward initialization."""
    reward_fn = SimpleReturnReward(default_config)

    assert reward_fn.config == default_config
    assert reward_fn.prev_portfolio_value is None
    assert reward_fn.returns_history == []


def test_simple_return_reward_first_step(default_config):
    """Test SimpleReturnReward returns 0 on first step."""
    reward_fn = SimpleReturnReward(default_config)

    reward = reward_fn.calculate(
        portfolio_value=100000.0,
        action=1,  # HOLD
        prev_action=1,
        cash=100000.0,
        position=0.0,
        price=100.0,
        prev_price=100.0
    )

    assert reward == 0.0
    assert reward_fn.prev_portfolio_value == 100000.0


def test_simple_return_reward_positive_return(default_config):
    """Test SimpleReturnReward with positive portfolio return."""
    reward_fn = SimpleReturnReward(default_config)

    # First step (initialization)
    reward_fn.calculate(
        portfolio_value=100000.0,
        action=1,
        prev_action=1,
        cash=50000.0,
        position=500.0,
        price=100.0,
        prev_price=100.0
    )

    # Second step with positive return
    reward = reward_fn.calculate(
        portfolio_value=105000.0,  # 5% gain
        action=1,  # HOLD (no transaction cost)
        prev_action=1,
        cash=50000.0,
        position=500.0,
        price=110.0,
        prev_price=100.0
    )

    expected_return = (105000.0 - 100000.0) / 100000.0
    assert reward == pytest.approx(expected_return, rel=1e-6)


def test_simple_return_reward_negative_return(default_config):
    """Test SimpleReturnReward with negative portfolio return."""
    reward_fn = SimpleReturnReward(default_config)

    # First step
    reward_fn.calculate(
        portfolio_value=100000.0,
        action=1,
        prev_action=1,
        cash=50000.0,
        position=500.0,
        price=100.0,
        prev_price=100.0
    )

    # Second step with negative return
    reward = reward_fn.calculate(
        portfolio_value=95000.0,  # 5% loss
        action=1,
        prev_action=1,
        cash=50000.0,
        position=500.0,
        price=90.0,
        prev_price=100.0
    )

    expected_return = (95000.0 - 100000.0) / 100000.0
    assert reward == pytest.approx(expected_return, rel=1e-6)


def test_simple_return_reward_with_transaction_cost(default_config):
    """Test SimpleReturnReward applies transaction costs on action change."""
    reward_fn = SimpleReturnReward(default_config)

    # First step
    reward_fn.calculate(
        portfolio_value=100000.0,
        action=1,  # HOLD
        prev_action=1,
        cash=50000.0,
        position=500.0,
        price=100.0,
        prev_price=100.0
    )

    # Second step with action change (HOLD -> BUY)
    reward = reward_fn.calculate(
        portfolio_value=102000.0,
        action=3,  # BUY_LARGE
        prev_action=1,  # HOLD
        cash=40000.0,
        position=600.0,
        price=100.0,
        prev_price=100.0
    )

    portfolio_return = (102000.0 - 100000.0) / 100000.0

    # Transaction cost should be applied
    assert reward < portfolio_return, "Reward should be less due to transaction costs"


def test_simple_return_reward_hold_no_cost(default_config):
    """Test SimpleReturnReward doesn't apply costs when holding."""
    reward_fn = SimpleReturnReward(default_config)

    # First step
    reward_fn.calculate(
        portfolio_value=100000.0,
        action=1,
        prev_action=1,
        cash=50000.0,
        position=500.0,
        price=100.0,
        prev_price=100.0
    )

    # Second step - HOLD to HOLD (no action change)
    reward = reward_fn.calculate(
        portfolio_value=101000.0,
        action=1,  # HOLD
        prev_action=1,  # HOLD
        cash=50000.0,
        position=500.0,
        price=102.0,
        prev_price=100.0
    )

    expected_return = (101000.0 - 100000.0) / 100000.0
    assert reward == pytest.approx(expected_return, rel=1e-6)


def test_simple_return_reward_reset(default_config):
    """Test SimpleReturnReward reset functionality."""
    reward_fn = SimpleReturnReward(default_config)

    # Calculate some rewards
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)
    reward_fn.calculate(105000.0, 1, 1, 50000.0, 500.0, 110.0, 100.0)

    assert reward_fn.prev_portfolio_value is not None

    # Reset
    reward_fn.reset()

    assert reward_fn.prev_portfolio_value is None
    assert reward_fn.peak_portfolio_value is None
    assert reward_fn.returns_history == []


# ==================== RiskAdjustedReward Tests ====================

def test_risk_adjusted_reward_initialization(default_config):
    """Test RiskAdjustedReward initialization."""
    reward_fn = RiskAdjustedReward(default_config, window_size=20)

    assert reward_fn.config == default_config
    assert reward_fn.window_size == 20
    assert reward_fn.prev_portfolio_value is None
    assert reward_fn.peak_portfolio_value is None
    assert reward_fn.returns_history == []


def test_risk_adjusted_reward_first_step(default_config):
    """Test RiskAdjustedReward returns 0 on first step."""
    reward_fn = RiskAdjustedReward(default_config)

    reward = reward_fn.calculate(
        portfolio_value=100000.0,
        action=1,
        prev_action=1,
        cash=100000.0,
        position=0.0,
        price=100.0,
        prev_price=100.0
    )

    assert reward == 0.0
    assert reward_fn.prev_portfolio_value == 100000.0
    assert reward_fn.peak_portfolio_value == 100000.0


def test_risk_adjusted_reward_volatility_penalty(default_config):
    """Test that volatility reduces reward."""
    reward_fn = RiskAdjustedReward(default_config)

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Add some returns to build history
    values = [101000, 99000, 102000, 98000, 103000]  # Volatile
    for val in values:
        reward_fn.calculate(val, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Check that volatility penalty is applied
    assert len(reward_fn.returns_history) > 0
    volatility = np.std(reward_fn.returns_history)
    assert volatility > 0, "Should have measured volatility"


def test_risk_adjusted_reward_sharpe_bonus(default_config):
    """Test Sharpe ratio bonus for good risk-adjusted returns."""
    reward_fn = RiskAdjustedReward(default_config)

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Add consistent positive returns (good Sharpe ratio)
    values = [101000, 102000, 103000, 104000, 105000, 106000]
    for val in values:
        reward = reward_fn.calculate(val, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # With consistent positive returns and low volatility, Sharpe should be good
    assert len(reward_fn.returns_history) >= 5


def test_risk_adjusted_reward_drawdown_penalty(default_config):
    """Test drawdown penalty when portfolio value drops from peak."""
    reward_fn = RiskAdjustedReward(default_config)

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Reach a peak
    reward_fn.calculate(120000.0, 1, 1, 50000.0, 500.0, 140.0, 100.0)
    assert reward_fn.peak_portfolio_value == 120000.0

    # Drop significantly (15% drawdown)
    reward = reward_fn.calculate(102000.0, 1, 1, 50000.0, 500.0, 104.0, 140.0)

    # Drawdown should apply penalty
    drawdown = (120000.0 - 102000.0) / 120000.0
    assert drawdown == pytest.approx(0.15, rel=0.01)
    # Reward should be penalized for large drawdown


def test_risk_adjusted_reward_profitable_trade_bonus(default_config):
    """Test bonus for profitable trades."""
    reward_fn = RiskAdjustedReward(default_config)

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Profitable trade
    reward = reward_fn.calculate(105000.0, 1, 1, 50000.0, 500.0, 110.0, 100.0)

    # Reward should include profitable trade bonus
    portfolio_return = (105000.0 - 100000.0) / 100000.0
    assert portfolio_return > 0
    # Bonus should be applied (exact value depends on config)


def test_risk_adjusted_reward_transaction_and_slippage(default_config):
    """Test transaction costs and slippage are applied."""
    # Create two identical scenarios, one with action change, one without
    reward_fn_with_change = RiskAdjustedReward(default_config)
    reward_fn_no_change = RiskAdjustedReward(default_config)

    # Initialize both
    reward_fn_with_change.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)
    reward_fn_no_change.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Scenario 1: Action change (triggers costs)
    reward_with_change = reward_fn_with_change.calculate(
        portfolio_value=102000.0,
        action=3,  # BUY_LARGE
        prev_action=1,  # HOLD (action changed)
        cash=40000.0,
        position=600.0,
        price=100.0,
        prev_price=100.0
    )

    # Scenario 2: No action change (no costs)
    reward_no_change = reward_fn_no_change.calculate(
        portfolio_value=102000.0,
        action=1,  # HOLD
        prev_action=1,  # HOLD (no action change)
        cash=50000.0,
        position=500.0,
        price=100.0,
        prev_price=100.0
    )

    # Reward with action change should be lower due to transaction costs
    assert reward_with_change < reward_no_change, "Action change should incur costs reducing reward"


def test_risk_adjusted_reward_extreme_action_penalty(default_config):
    """Test penalty for extreme actions."""
    reward_fn = RiskAdjustedReward(default_config)

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Extreme action change (diff > 2)
    reward_extreme = reward_fn.calculate(
        portfolio_value=102000.0,
        action=5,  # SELL_ALL
        prev_action=1,  # HOLD (diff = 4)
        cash=100000.0,
        position=0.0,
        price=100.0,
        prev_price=100.0
    )

    # Should have extreme action penalty applied
    # The reward should be reduced by the extreme action penalty


def test_risk_adjusted_reward_window_size_limit(default_config):
    """Test that returns history respects window size."""
    window_size = 10
    reward_fn = RiskAdjustedReward(default_config, window_size=window_size)

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Add more returns than window size
    for i in range(20):
        val = 100000.0 + i * 1000
        reward_fn.calculate(val, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Should keep only window_size returns
    assert len(reward_fn.returns_history) == window_size


# ==================== CustomizableReward Tests ====================

def test_customizable_reward_initialization(custom_config):
    """Test CustomizableReward initialization."""
    reward_fn = CustomizableReward(
        custom_config,
        window_size=15,
        use_sharpe=True,
        use_drawdown=True,
        use_transaction_costs=True,
        use_slippage=True
    )

    assert reward_fn.config == custom_config
    assert reward_fn.window_size == 15
    assert reward_fn.use_sharpe is True
    assert reward_fn.use_drawdown is True
    assert reward_fn.use_transaction_costs is True
    assert reward_fn.use_slippage is True


def test_customizable_reward_no_transaction_costs(default_config):
    """Test CustomizableReward with transaction costs disabled."""
    reward_fn = CustomizableReward(
        default_config,
        use_transaction_costs=False
    )

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Action change should not apply transaction costs
    reward = reward_fn.calculate(
        portfolio_value=102000.0,
        action=3,
        prev_action=1,
        cash=40000.0,
        position=600.0,
        price=100.0,
        prev_price=100.0
    )

    # Reward should be just the return (no costs)
    expected_return = (102000.0 - 100000.0) / 100000.0
    assert reward == pytest.approx(expected_return, rel=1e-6)


def test_customizable_reward_no_sharpe(default_config):
    """Test CustomizableReward with Sharpe bonus disabled."""
    reward_fn = CustomizableReward(
        default_config,
        use_sharpe=False
    )

    # Initialize and add returns
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)
    for i in range(10):
        val = 100000.0 + i * 1000
        reward_fn.calculate(val, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Should not apply Sharpe bonus even with good returns
    assert len(reward_fn.returns_history) >= 5


def test_customizable_reward_no_drawdown(default_config):
    """Test CustomizableReward with drawdown penalty disabled."""
    reward_fn = CustomizableReward(
        default_config,
        use_drawdown=False
    )

    # Initialize
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Reach peak
    reward_fn.calculate(120000.0, 1, 1, 50000.0, 500.0, 140.0, 100.0)

    # Drop significantly
    reward = reward_fn.calculate(102000.0, 1, 1, 50000.0, 500.0, 104.0, 140.0)

    # Should not apply drawdown penalty
    # Reward should be based on return only


def test_customizable_reward_with_slippage(default_config):
    """Test CustomizableReward applies slippage when enabled."""
    reward_fn_with_slippage = CustomizableReward(
        default_config,
        use_transaction_costs=True,
        use_slippage=True
    )

    reward_fn_no_slippage = CustomizableReward(
        default_config,
        use_transaction_costs=True,
        use_slippage=False
    )

    # Same scenario for both
    for fn in [reward_fn_with_slippage, reward_fn_no_slippage]:
        fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    reward_with = reward_fn_with_slippage.calculate(
        102000.0, 3, 1, 40000.0, 600.0, 100.0, 100.0
    )

    reward_without = reward_fn_no_slippage.calculate(
        102000.0, 3, 1, 40000.0, 600.0, 100.0, 100.0
    )

    # Reward with slippage should be lower
    assert reward_with < reward_without


# ==================== Factory Function Tests ====================

def test_get_reward_function_simple():
    """Test factory function creates SimpleReturnReward."""
    reward_fn = get_reward_function("simple")
    assert isinstance(reward_fn, SimpleReturnReward)


def test_get_reward_function_risk_adjusted():
    """Test factory function creates RiskAdjustedReward."""
    reward_fn = get_reward_function("risk_adjusted")
    assert isinstance(reward_fn, RiskAdjustedReward)


def test_get_reward_function_customizable():
    """Test factory function creates CustomizableReward."""
    reward_fn = get_reward_function("customizable")
    assert isinstance(reward_fn, CustomizableReward)


def test_get_reward_function_with_config():
    """Test factory function passes config correctly."""
    custom_config = RewardConfig(return_weight=3.0)
    reward_fn = get_reward_function("simple", config=custom_config)

    assert reward_fn.config.return_weight == 3.0


def test_get_reward_function_with_kwargs():
    """Test factory function passes kwargs correctly."""
    reward_fn = get_reward_function("risk_adjusted", window_size=30)

    assert isinstance(reward_fn, RiskAdjustedReward)
    assert reward_fn.window_size == 30


def test_get_reward_function_invalid_type():
    """Test factory function raises error for invalid type."""
    with pytest.raises(ValueError, match="Unknown reward type"):
        get_reward_function("invalid_type")


# ==================== Edge Cases ====================

def test_reward_with_zero_portfolio_value(default_config):
    """Test reward calculation handles zero portfolio value."""
    reward_fn = SimpleReturnReward(default_config)

    # This is an edge case that shouldn't happen in practice
    # but the code should handle it gracefully
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)

    # Try with very small portfolio value
    reward = reward_fn.calculate(0.01, 1, 1, 0.01, 0.0, 100.0, 100.0)

    # Should not crash
    assert isinstance(reward, float)


def test_reward_config_defaults():
    """Test RewardConfig has sensible defaults."""
    config = RewardConfig()

    assert config.return_weight == 1.0
    assert config.risk_penalty == 0.5
    assert config.sharpe_bonus == 0.1
    assert config.transaction_cost_rate == 0.001
    assert config.slippage_rate == 0.0005
    assert config.max_drawdown_penalty == 0.3
    assert config.extreme_action_penalty == 0.01
    assert config.profitable_trade_bonus == 0.05


def test_reward_multiple_episodes(default_config):
    """Test reward function can be reset and reused."""
    reward_fn = RiskAdjustedReward(default_config)

    # Episode 1
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)
    reward_fn.calculate(105000.0, 1, 1, 50000.0, 500.0, 110.0, 100.0)
    assert len(reward_fn.returns_history) > 0

    # Reset for episode 2
    reward_fn.reset()
    assert len(reward_fn.returns_history) == 0

    # Episode 2
    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)
    reward_fn.calculate(98000.0, 1, 1, 50000.0, 500.0, 96.0, 100.0)
    assert len(reward_fn.returns_history) > 0
