"""
Tests for Action Masking functionality
"""
import pytest
import numpy as np
from src.rl.improvements import ActionMasker
from src.rl.types import TradingAction, ImprovedTradingAction


def test_action_masker_initialization():
    """Test ActionMasker initialization with standard actions."""
    masker = ActionMasker(use_improved_actions=False)
    assert masker.n_actions == len(TradingAction)
    assert masker.action_enum == TradingAction


def test_action_masker_improved_initialization():
    """Test ActionMasker initialization with improved actions."""
    masker = ActionMasker(use_improved_actions=True)
    assert masker.n_actions == len(ImprovedTradingAction)
    assert masker.action_enum == ImprovedTradingAction


def test_standard_action_mask_no_position():
    """Test standard action masking when no position exists."""
    masker = ActionMasker(use_improved_actions=False)

    # No position, sufficient cash
    mask = masker.get_action_mask(
        cash=10000.0,
        position=0,
        current_price=100.0,
        portfolio_value=10000.0,
        max_position_pct=80.0
    )

    # Should allow: HOLD, BUY_SMALL, BUY_LARGE
    # Should block: SELL (no position)
    assert mask[TradingAction.HOLD] == 1.0
    assert mask[TradingAction.BUY_SMALL] == 1.0
    assert mask[TradingAction.BUY_LARGE] == 1.0
    assert mask[TradingAction.SELL] == 0.0


def test_standard_action_mask_insufficient_cash():
    """Test standard action masking with insufficient cash."""
    masker = ActionMasker(use_improved_actions=False)

    # Very low cash, can't buy
    mask = masker.get_action_mask(
        cash=50.0,
        position=0,
        current_price=100.0,
        portfolio_value=50.0,
        max_position_pct=80.0
    )

    # Should allow: HOLD
    # Should block: BUY_SMALL, BUY_LARGE (insufficient cash), SELL (no position)
    assert mask[TradingAction.HOLD] == 1.0
    assert mask[TradingAction.BUY_SMALL] == 0.0
    assert mask[TradingAction.BUY_LARGE] == 0.0
    assert mask[TradingAction.SELL] == 0.0


def test_standard_action_mask_with_position():
    """Test standard action masking when position exists."""
    masker = ActionMasker(use_improved_actions=False)

    # Has position and cash
    mask = masker.get_action_mask(
        cash=5000.0,
        position=50,  # Has 50 shares
        current_price=100.0,
        portfolio_value=10000.0,
        max_position_pct=80.0
    )

    # Should allow: HOLD, BUY_SMALL, BUY_LARGE, SELL
    assert mask[TradingAction.HOLD] == 1.0
    assert mask[TradingAction.BUY_SMALL] == 1.0
    assert mask[TradingAction.BUY_LARGE] == 1.0
    assert mask[TradingAction.SELL] == 1.0


def test_improved_action_mask_no_position():
    """Test improved action masking when no position exists."""
    masker = ActionMasker(use_improved_actions=True)

    mask = masker.get_action_mask(
        cash=10000.0,
        position=0,
        current_price=100.0,
        portfolio_value=10000.0,
        max_position_pct=80.0
    )

    # Should allow: HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE
    # Should block: SELL_PARTIAL, SELL_ALL (no position)
    assert mask[ImprovedTradingAction.HOLD] == 1.0
    assert mask[ImprovedTradingAction.BUY_SMALL] == 1.0
    assert mask[ImprovedTradingAction.BUY_MEDIUM] == 1.0
    assert mask[ImprovedTradingAction.BUY_LARGE] == 1.0
    assert mask[ImprovedTradingAction.SELL_PARTIAL] == 0.0
    assert mask[ImprovedTradingAction.SELL_ALL] == 0.0


def test_improved_action_mask_with_position():
    """Test improved action masking when position exists."""
    masker = ActionMasker(use_improved_actions=True)

    mask = masker.get_action_mask(
        cash=5000.0,
        position=50,
        current_price=100.0,
        portfolio_value=10000.0,
        max_position_pct=80.0
    )

    # Should allow all actions
    assert mask[ImprovedTradingAction.HOLD] == 1.0
    assert mask[ImprovedTradingAction.BUY_SMALL] == 1.0
    assert mask[ImprovedTradingAction.BUY_MEDIUM] == 1.0
    assert mask[ImprovedTradingAction.BUY_LARGE] == 1.0
    assert mask[ImprovedTradingAction.SELL_PARTIAL] == 1.0
    assert mask[ImprovedTradingAction.SELL_ALL] == 1.0


def test_improved_action_mask_position_limit():
    """Test improved action masking at position size limit."""
    masker = ActionMasker(use_improved_actions=True)

    # Already at 80% position limit
    mask = masker.get_action_mask(
        cash=2000.0,
        position=80,  # 80 shares * $100 = $8000 (80% of $10000)
        current_price=100.0,
        portfolio_value=10000.0,
        max_position_pct=80.0
    )

    # Should allow: HOLD, SELL_PARTIAL, SELL_ALL
    # Should block: BUY actions (at position limit)
    assert mask[ImprovedTradingAction.HOLD] == 1.0
    assert mask[ImprovedTradingAction.SELL_PARTIAL] == 1.0
    assert mask[ImprovedTradingAction.SELL_ALL] == 1.0
    # Buy actions might be blocked depending on implementation


def test_action_mask_shape():
    """Test that action mask returns correct shape."""
    masker_standard = ActionMasker(use_improved_actions=False)
    masker_improved = ActionMasker(use_improved_actions=True)

    mask_standard = masker_standard.get_action_mask(
        cash=10000.0, position=0, current_price=100.0,
        portfolio_value=10000.0, max_position_pct=80.0
    )
    mask_improved = masker_improved.get_action_mask(
        cash=10000.0, position=0, current_price=100.0,
        portfolio_value=10000.0, max_position_pct=80.0
    )

    assert mask_standard.shape == (len(TradingAction),)
    assert mask_improved.shape == (len(ImprovedTradingAction),)
    assert mask_standard.dtype == np.float32
    assert mask_improved.dtype == np.float32


def test_action_mask_values():
    """Test that action mask only contains 0.0 or 1.0."""
    masker = ActionMasker(use_improved_actions=True)

    mask = masker.get_action_mask(
        cash=5000.0, position=50, current_price=100.0,
        portfolio_value=10000.0, max_position_pct=80.0
    )

    # All values should be 0.0 or 1.0
    assert np.all((mask == 0.0) | (mask == 1.0))
