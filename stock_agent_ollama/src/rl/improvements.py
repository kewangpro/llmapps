"""
Comprehensive RL Training Improvements

This module contains enhanced components for better RL training and live trading:
1. Action masking to prevent invalid actions
2. Advanced reward shaping
3. Adaptive action sizing
4. Curriculum learning
5. Enhanced observation space
6. Training diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum
import logging

from .types import TradingAction, ImprovedTradingAction
from ..config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# ACTION MASKING
# ============================================================================

class ActionMasker:
    """
    Prevents the agent from taking invalid actions.

    Invalid actions include:
    - Selling when no position exists
    - Buying when insufficient cash
    - Exceeding position size limits
    """

    def __init__(self, use_improved_actions: bool = False):
        self.use_improved_actions = use_improved_actions
        self.action_enum = ImprovedTradingAction if use_improved_actions else TradingAction
        self.n_actions = len(self.action_enum)

    def get_action_mask(
        self,
        cash: float,
        position: int,
        current_price: float,
        portfolio_value: float,
        max_position_pct: float = 80.0
    ) -> np.ndarray:
        """
        Get binary mask of valid actions (1=valid, 0=invalid).

        Args:
            cash: Available cash
            position: Current position size (shares)
            current_price: Current stock price
            portfolio_value: Total portfolio value
            max_position_pct: Max position as % of portfolio value

        Returns:
            Binary mask array
        """
        mask = np.ones(self.n_actions, dtype=np.float32)

        if self.use_improved_actions:
            return self._get_improved_action_mask(
                cash, position, current_price,
                portfolio_value, max_position_pct
            )
        else:
            return self._get_standard_action_mask(
                cash, position, current_price,
                portfolio_value, max_position_pct
            )

    def _get_standard_action_mask(
        self, cash: float, position: int, current_price: float,
        portfolio_value: float, max_position_pct: float
    ) -> np.ndarray:
        """Mask for standard TradingAction space."""
        mask = np.ones(len(TradingAction), dtype=np.float32)

        # Can't sell if no position
        if position == 0:
            mask[TradingAction.SELL] = 0.0

        # Check if can buy at least 1 share
        min_buy_cost = current_price * 1.01  # Include small buffer for costs

        # BUY_SMALL (10% of cash)
        buy_small_amount = cash * 0.1
        if buy_small_amount < min_buy_cost:
            mask[TradingAction.BUY_SMALL] = 0.0
        else:
            # Check percentage limit only
            shares_small = int(buy_small_amount / current_price)
            new_position_value = (position + shares_small) * current_price
            if (new_position_value / portfolio_value * 100) > max_position_pct:
                mask[TradingAction.BUY_SMALL] = 0.0

        # BUY_LARGE (30% of cash)
        buy_large_amount = cash * 0.3
        if buy_large_amount < min_buy_cost:
            mask[TradingAction.BUY_LARGE] = 0.0
        else:
            # Check percentage limit only
            shares_large = int(buy_large_amount / current_price)
            new_position_value = (position + shares_large) * current_price
            if (new_position_value / portfolio_value * 100) > max_position_pct:
                mask[TradingAction.BUY_LARGE] = 0.0

        # Ensure at least HOLD is always valid
        mask[TradingAction.HOLD] = 1.0

        return mask

    def _get_improved_action_mask(
        self, cash: float, position: int, current_price: float,
        portfolio_value: float, max_position_pct: float
    ) -> np.ndarray:
        """Mask for improved ImprovedTradingAction space."""
        mask = np.ones(len(ImprovedTradingAction), dtype=np.float32)

        # HOLD is always valid
        mask[ImprovedTradingAction.HOLD] = 1.0

        # Can't sell if no position
        if position == 0:
            mask[ImprovedTradingAction.SELL_PARTIAL] = 0.0
            mask[ImprovedTradingAction.SELL_ALL] = 0.0
        elif position < 2:
            # Can't sell partial if only 1 share
            mask[ImprovedTradingAction.SELL_PARTIAL] = 0.0

        # Check buy actions
        min_buy_cost = current_price * 1.01

        for action in [ImprovedTradingAction.BUY_SMALL,
                       ImprovedTradingAction.BUY_MEDIUM,
                       ImprovedTradingAction.BUY_LARGE]:
            # Determine buy amount
            if action == ImprovedTradingAction.BUY_SMALL:
                buy_pct = 0.15  # 15% average
            elif action == ImprovedTradingAction.BUY_MEDIUM:
                buy_pct = 0.30  # 30% average
            else:
                buy_pct = 0.50  # 50% average

            buy_amount = cash * buy_pct

            if buy_amount < min_buy_cost:
                mask[action] = 0.0
            else:
                shares = int(buy_amount / current_price)
                new_position_value = (position + shares) * current_price
                position_pct = (new_position_value / portfolio_value * 100) if portfolio_value > 0 else 0

                # Only check percentage-based position limit (removed share count limit)
                if position_pct > max_position_pct:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Masking {action.name}: new position {position_pct:.1f}% would exceed max_position_pct {max_position_pct:.1f}% (current={position} shares, buying {shares} @ ${current_price:.2f}, cash=${cash:.2f}, portfolio=${portfolio_value:.2f})")
                    mask[action] = 0.0

        return mask

    def apply_mask_to_logits(self, action_logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply action mask to logits by setting invalid actions to -inf.

        Args:
            action_logits: Raw action logits from model
            mask: Binary mask (1=valid, 0=invalid)

        Returns:
            Masked logits
        """
        # Set invalid actions to very negative value
        masked_logits = action_logits.copy()
        masked_logits[mask == 0] = -1e10
        return masked_logits


# ============================================================================
# ENHANCED REWARD FUNCTION
# ============================================================================

@dataclass
class EnhancedRewardConfig:
    """
    Configuration for enhanced reward function.

    OPTIMIZED FOR QRDQN - Base reward configuration.
    Lighter penalties allow exploration and learning from Q-values.

    For PPO, use PPORewardConfig instead.
    For SAC, use SACRewardConfig instead.
    For RecurrentPPO, use RecurrentPPORewardConfig.
    """
    # Base rewards
    return_weight: float = 1.0

    # Penalties
    # Balanced to discourage invalid actions without causing action collapse
    invalid_action_penalty: float = -1.0
    # INCREASED to discourage high-frequency "churning" behavior in live trading
    # Models trained on daily data should not trade every minute
    excessive_trading_penalty: float = -0.5  # Increased from -0.1
    # QRDQN/SAC work well with lighter penalties (learn risk naturally via Q-values)
    risk_penalty_weight: float = 0.01
    drawdown_penalty_weight: float = 0.05

    # Bonuses
    profitable_trade_bonus: float = 0.1
    sharpe_bonus_weight: float = 0.2
    valid_action_bonus: float = 0.1

    # Transaction costs
    # QRDQN/SAC learn from costs naturally, weak penalty allows optimal trading frequency
    transaction_cost_rate: float = 0.0005
    slippage_rate: float = 0.0005

    # Shaping parameters
    use_action_shaping: bool = True
    use_progress_shaping: bool = True
    use_risk_shaping: bool = True

    # Advanced
    min_hold_steps: int = 5  # Minimum steps to hold before selling

    # Legacy fields (not used in current implementation)
    action_diversity_bonus: float = 0.1  # NOT USED
    hold_winner_bonus: float = 0.1  # NOT USED
    diversity_window: int = 50  # NOT USED


@dataclass
class PPORewardConfig(EnhancedRewardConfig):
    """
    Reward configuration optimized for PPO.

    PPO needs stronger penalties/bonuses because:
    - Entropy-based exploration (vs epsilon-greedy)
    - On-policy learning (discards experiences)
    - Prone to action collapse

    These stronger values help prevent collapse.
    """
    # Stronger penalties to discourage collapse
    risk_penalty_weight: float = 0.3
    drawdown_penalty_weight: float = 0.5

    # Stronger diversity bonus to prevent action collapse
    action_diversity_bonus: float = 1.0

    # Higher transaction costs to discourage overtrading
    transaction_cost_rate: float = 0.002


@dataclass
class RecurrentPPORewardConfig(PPORewardConfig):
    """
    Reward configuration optimized for RecurrentPPO.

    RecurrentPPO uses LSTM memory with trend indicators for temporal pattern recognition.
    This config encourages trend-following behavior:
    - Moderate HOLD incentive during uptrends
    - Momentum trend bonus for riding winners
    - Balanced penalties to avoid premature exits but protect gains
    - Encourages fuller position sizing
    """
    # Balanced penalties to allow trend riding while protecting gains
    risk_penalty_weight: float = 0.1  # Increased from 0.05
    drawdown_penalty_weight: float = 0.2  # Increased from 0.1

    # Moderate HOLD incentive during winning positions
    # (Reduced from 0.5 to prevent being 'too sticky' to positions)
    hold_winning_position_bonus: float = 0.15

    # Momentum bonus for staying invested during uptrends
    momentum_trend_bonus: float = 0.2  # Reduced from 0.3

    # INCREASED to prevent high-frequency trading with daily-trained models
    # RecurrentPPO with LSTM should learn to hold positions, not churn
    excessive_trading_penalty: float = -0.3

    # Strong reward for profitable trades
    profitable_trade_bonus: float = 0.3  # Reduced from 0.5

    # Minimal transaction costs
    transaction_cost_rate: float = 0.001  # Further reduced


class EnhancedRewardFunction:
    """
    Advanced reward function with:
    - Invalid action penalties
    - Action sequence shaping
    - Risk-adjusted returns
    - Progressive difficulty
    """

    def __init__(
        self,
        config: Optional[EnhancedRewardConfig] = None,
        action_masker: Optional[ActionMasker] = None,
        use_improved_actions: bool = False
    ):
        self.config = config or EnhancedRewardConfig()
        self.action_masker = action_masker or ActionMasker(use_improved_actions)
        self.use_improved_actions = use_improved_actions

        # State tracking
        self.prev_portfolio_value = None
        self.peak_portfolio_value = None
        self.returns_history = []
        self.window_size = 20

        # Action tracking
        self.last_buy_step = None
        self.last_sell_step = None
        self.consecutive_same_actions = 0
        self.prev_action = None
        self.step_count = 0
        self.recent_actions = []  # For action diversity tracking

    def reset(self):
        """Reset internal state."""
        self.prev_portfolio_value = None
        self.peak_portfolio_value = None
        self.returns_history = []
        self.last_buy_step = None
        self.last_sell_step = None
        self.consecutive_same_actions = 0
        self.prev_action = None
        self.step_count = 0
        self.recent_actions = []  # Reset action diversity tracking

    def calculate(
        self,
        portfolio_value: float,
        action: int,
        prev_action: int,
        cash: float,
        position: float,
        price: float,
        prev_price: float,
        max_position_pct: float = 80.0,
        **kwargs
    ) -> float:
        """
        Calculate enhanced reward with multiple components.

        Args:
            portfolio_value: Current portfolio value
            action: Action taken
            prev_action: Previous action
            cash: Current cash
            position: Current position (shares)
            price: Current price
            prev_price: Previous price
            max_position_pct: Max position as % of portfolio

        Returns:
            Calculated reward
        """
        self.step_count += 1

        # Initialize on first step
        if self.prev_portfolio_value is None:
            self.prev_portfolio_value = portfolio_value
            self.peak_portfolio_value = portfolio_value
            self.prev_action = action
            return 0.0

        # Calculate base return
        portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.returns_history.append(portfolio_return)

        # Keep only recent history
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Start with portfolio return
        reward = portfolio_return * self.config.return_weight

        # === 1. INVALID ACTION PENALTY ===
        if self.config.use_action_shaping:
            action_mask = self.action_masker.get_action_mask(
                cash=cash,
                position=int(position),
                current_price=price,
                portfolio_value=portfolio_value,
                max_position_pct=max_position_pct
            )

            # Penalize if action was invalid
            if action_mask[action] == 0:
                reward += self.config.invalid_action_penalty
                logger.debug(f"Invalid action {action} penalized")
            else:
                # Small bonus for valid action
                reward += self.config.valid_action_bonus

        # === 2. ACTION SEQUENCE SHAPING ===
        if self.config.use_action_shaping:
            # Penalize excessive trading (same action repeatedly)
            # FIX 2.0: Start penalty IMMEDIATELY on first repeat (not 2nd!)
            is_trading_action = action != (ImprovedTradingAction.HOLD if self.use_improved_actions
                                          else TradingAction.HOLD)

            if action == self.prev_action and is_trading_action:
                self.consecutive_same_actions += 1

                # Progressive penalty: START AT FIRST REPEAT
                if self.consecutive_same_actions >= 1:  # Changed from >= 2
                    # Scale penalty: 1x, 2x, 3x, 4x, 5x (cap at 5x)
                    penalty_multiplier = min(self.consecutive_same_actions, 5)  # Removed -1
                    progressive_penalty = self.config.excessive_trading_penalty * penalty_multiplier
                    reward += progressive_penalty  # excessive_trading_penalty is negative
                    logger.debug(f"Consecutive action penalty ({self.consecutive_same_actions}x): {progressive_penalty:.4f}")
            else:
                self.consecutive_same_actions = 0

            # Encourage holding after buying
            if self.use_improved_actions:
                is_buy = action in [1, 2, 3]  # BUY_SMALL, BUY_MEDIUM, BUY_LARGE
                is_sell = action in [4, 5]    # SELL_PARTIAL, SELL_ALL
            else:
                is_buy = action in [2, 3]     # BUY_SMALL, BUY_LARGE
                is_sell = action == 0         # SELL

            if is_buy:
                self.last_buy_step = self.step_count

            if is_sell and self.last_buy_step is not None:
                steps_held = self.step_count - self.last_buy_step
                if steps_held < self.config.min_hold_steps:
                    # Penalize selling too quickly
                    reward -= 0.1 * (self.config.min_hold_steps - steps_held) / self.config.min_hold_steps

        # === 3. TRANSACTION COSTS ===
        # FIX: Apply costs to ALL trades (buy or sell), not just when action changes
        # This prevents SAC from spamming BUY_MEDIUM repeatedly with zero cost
        if self.use_improved_actions:
            is_buy = action in [1, 2, 3]  # BUY_SMALL, BUY_MEDIUM, BUY_LARGE
            is_sell = action in [4, 5]    # SELL_PARTIAL, SELL_ALL
        else:
            is_buy = action in [2, 3]     # BUY_SMALL, BUY_LARGE (old action space)
            is_sell = action == 0         # SELL (old action space)

        is_trading = is_buy or is_sell

        if is_trading:  # Charge costs for every trade execution
            transaction_value = price * max(position, 100)  # Estimate trade value
            transaction_cost = transaction_value * self.config.transaction_cost_rate
            slippage = transaction_value * self.config.slippage_rate
            total_cost = (transaction_cost + slippage) / self.prev_portfolio_value
            reward -= total_cost
            logger.debug(f"Transaction costs applied: -{total_cost:.6f}")

        # === 4. RISK PENALTIES ===
        if self.config.use_risk_shaping and len(self.returns_history) >= 2:
            # Volatility penalty
            volatility = np.std(self.returns_history)
            reward -= volatility * self.config.risk_penalty_weight

            # Sharpe bonus
            if len(self.returns_history) >= 5:
                mean_return = np.mean(self.returns_history)
                sharpe = mean_return / (volatility + 1e-8)
                if sharpe > 0:
                    reward += sharpe * self.config.sharpe_bonus_weight

        # === 5. DRAWDOWN PENALTY ===
        if self.config.use_risk_shaping:
            self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
            drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value

            if drawdown > 0.05:  # More than 5% drawdown
                reward -= drawdown * self.config.drawdown_penalty_weight

        # === 6. PROFITABLE TRADE BONUS ===
        if portfolio_return > 0:
            reward += self.config.profitable_trade_bonus

        # === 7. HOLD INCENTIVES ===
        # Encourage holding winning positions (especially for RecurrentPPO and SAC)
        if self.config.use_action_shaping and hasattr(self.config, 'hold_winning_position_bonus'):
            is_hold = (action == ImprovedTradingAction.HOLD if self.use_improved_actions
                      else action == TradingAction.HOLD)

            # HOLD winner bonus: extra reward for holding profitable positions
            if is_hold and position > 0 and portfolio_return > 0:
                reward += self.config.hold_winning_position_bonus
                logger.debug(f"HOLD winning position bonus: +{self.config.hold_winning_position_bonus}")

            # FIX: Base HOLD incentive (always reward HOLD to encourage exploration)
            # This ensures SAC explores HOLD action even when not in winning position
            if is_hold and hasattr(self.config, 'base_hold_incentive'):
                reward += self.config.base_hold_incentive
                logger.debug(f"Base HOLD incentive: +{self.config.base_hold_incentive}")

        # === 8. MOMENTUM TREND BONUS ===
        # Bonus for holding positions during strong upward momentum
        if self.config.use_action_shaping and hasattr(self.config, 'momentum_trend_bonus'):
            # Calculate price momentum
            price_momentum = (price - prev_price) / prev_price if prev_price > 0 else 0

            # Calculate portfolio momentum (recent trend)
            recent_returns = self.returns_history[-5:] if len(self.returns_history) >= 5 else self.returns_history
            avg_recent_return = np.mean(recent_returns) if len(recent_returns) > 0 else 0

            # Strong uptrend: positive price momentum AND positive recent returns AND holding position
            is_strong_uptrend = price_momentum > 0.005 and avg_recent_return > 0 and position > 0

            # Bonus for holding or buying during uptrend (not selling)
            is_hold_or_buy = (action == ImprovedTradingAction.HOLD or
                             action == ImprovedTradingAction.BUY_SMALL or
                             action == ImprovedTradingAction.BUY_MEDIUM or
                             action == ImprovedTradingAction.BUY_LARGE) if self.use_improved_actions else (
                             action == TradingAction.HOLD or action == TradingAction.BUY)

            if is_strong_uptrend and is_hold_or_buy:
                reward += self.config.momentum_trend_bonus
                logger.debug(f"Momentum trend bonus applied: +{self.config.momentum_trend_bonus}")

        # === 9. ACTION DIVERSITY REWARD/PENALTY ===
        # FIX v5: Smaller, more reactive window to catch collapse early
        if hasattr(self.config, 'diversity_bonus'):
            # Track recent actions (smaller window for immediate feedback)
            self.recent_actions.append(action)
            if len(self.recent_actions) > 20:  # Keep last 20 actions for balance
                self.recent_actions.pop(0)

            # Calculate diversity ratio (very early check to prevent initial collapse)
            if len(self.recent_actions) >= 5:  # Check after just 5 actions
                unique_actions = len(set(self.recent_actions))
                diversity_ratio = unique_actions / len(self.recent_actions)

                # AGGRESSIVE: Penalize low diversity (<30%), reward high diversity (>50%)
                if diversity_ratio < 0.3:
                    # Severe penalty for action collapse
                    if hasattr(self.config, 'diversity_penalty'):
                        diversity_punishment = self.config.diversity_penalty * (1.0 - diversity_ratio)
                        reward += diversity_punishment  # diversity_penalty is negative
                        logger.warning(f"⚠️  ACTION COLLAPSE DETECTED: {diversity_punishment:.4f} penalty "
                                      f"({unique_actions}/{len(self.recent_actions)} = {diversity_ratio:.1%})")
                elif diversity_ratio > 0.5:
                    # Reward good diversity
                    diversity_reward = self.config.diversity_bonus * diversity_ratio
                    reward += diversity_reward
                    logger.debug(f"✅ Action diversity reward: +{diversity_reward:.4f} "
                               f"({unique_actions}/{len(self.recent_actions)} = {diversity_ratio:.1%} unique)")

        # Update state
        self.prev_portfolio_value = portfolio_value
        self.prev_action = action

        return reward


# ============================================================================
# ADAPTIVE ACTION SIZING
# ============================================================================

class AdaptiveActionSizer:
    """
    Dynamically adjusts buy/sell sizes based on:
    - Available cash
    - Portfolio concentration
    - Market volatility
    - Model confidence
    """

    def __init__(
        self,
        min_buy_pct: float = 0.05,
        max_buy_pct: float = 0.60,
        min_shares: int = 1
    ):
        self.min_buy_pct = min_buy_pct
        self.max_buy_pct = max_buy_pct
        self.min_shares = min_shares

    def get_buy_size(
        self,
        action: int,
        cash: float,
        price: float,
        position: int,
        portfolio_value: float,
        max_position_pct: float,
        volatility: float = 0.02,
        use_improved_actions: bool = False
    ) -> int:
        """
        Calculate adaptive buy size in shares.

        Args:
            action: The buy action
            cash: Available cash
            price: Current price
            position: Current position
            portfolio_value: Total portfolio value
            max_position_pct: Max position as % of portfolio
            volatility: Recent volatility (for sizing adjustment)
            use_improved_actions: Whether using improved action space

        Returns:
            Number of shares to buy
        """
        # Determine base percentage based on action
        if use_improved_actions:
            if action == ImprovedTradingAction.BUY_SMALL:
                base_pct = 0.15
            elif action == ImprovedTradingAction.BUY_MEDIUM:
                base_pct = 0.30
            elif action == ImprovedTradingAction.BUY_LARGE:
                base_pct = 0.50
            else:
                return 0
        else:
            if action == TradingAction.BUY_SMALL:
                base_pct = 0.10
            elif action == TradingAction.BUY_LARGE:
                base_pct = 0.30
            else:
                return 0

        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = 1.0 - min(volatility * 10, 0.5)
        adjusted_pct = base_pct * volatility_adjustment

        # Ensure within bounds
        adjusted_pct = np.clip(adjusted_pct, self.min_buy_pct, self.max_buy_pct)

        # Calculate shares from cash percentage
        buy_amount = cash * adjusted_pct
        affordable_shares = int(buy_amount / price)

        # Check position limit
        current_position_value = position * price
        max_position_value = portfolio_value * (max_position_pct / 100.0)
        remaining_capacity = max_position_value - current_position_value
        max_additional_shares = int(remaining_capacity / price)

        # Take minimum of affordable and capacity
        shares = min(affordable_shares, max_additional_shares)

        # Ensure at least min_shares if we're buying
        if shares < self.min_shares and affordable_shares >= self.min_shares:
            shares = self.min_shares

        return max(0, shares)

    def get_sell_size(
        self,
        action: int,
        position: int,
        use_improved_actions: bool = False
    ) -> int:
        """
        Calculate sell size in shares.

        Args:
            action: The sell action
            position: Current position
            use_improved_actions: Whether using improved action space

        Returns:
            Number of shares to sell (0 if no position)
        """
        # Can't sell if no position
        if position == 0:
            return 0

        if use_improved_actions:
            if action == ImprovedTradingAction.SELL_PARTIAL:
                # Sell 50% of position, minimum 1 share if position > 0
                return max(1, position // 2)
            elif action == ImprovedTradingAction.SELL_ALL:
                return position
        else:
            if action == TradingAction.SELL:
                return position

        return 0


# ============================================================================
# CURRICULUM LEARNING
# ============================================================================

@dataclass
class CurriculumStage:
    """A single stage in curriculum learning."""
    name: str
    min_episodes: int
    difficulty: float  # 0.0 to 1.0
    description: str

    # Stage-specific parameters
    start_with_position: bool = False
    initial_position_pct: float = 0.0
    allow_sell_only: bool = False
    reduced_action_space: bool = False


class CurriculumManager:
    """
    Manages curriculum learning progression:
    Stage 1: Learn to HOLD and basic observation
    Stage 2: Learn to BUY (start with cash)
    Stage 3: Learn to SELL (start with position)
    Stage 4: Learn full sequences (start with cash)
    Stage 5: Advanced scenarios (varied starts)
    """

    def __init__(self):
        self.current_stage = 0
        self.episode_count = 0
        self.stages = [
            CurriculumStage(
                name="Foundation",
                min_episodes=50,
                difficulty=0.2,
                description="Learn to observe and hold",
                start_with_position=False,
                reduced_action_space=True
            ),
            CurriculumStage(
                name="Buying",
                min_episodes=100,
                difficulty=0.4,
                description="Learn when to buy",
                start_with_position=False,
                initial_position_pct=0.0
            ),
            CurriculumStage(
                name="Selling",
                min_episodes=100,
                difficulty=0.6,
                description="Learn when to sell",
                start_with_position=True,
                initial_position_pct=0.3
            ),
            CurriculumStage(
                name="Sequences",
                min_episodes=200,
                difficulty=0.8,
                description="Learn buy-hold-sell sequences",
                start_with_position=False
            ),
            CurriculumStage(
                name="Advanced",
                min_episodes=300,
                difficulty=1.0,
                description="Master all scenarios",
                start_with_position=False  # Random starts
            )
        ]

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]

    def should_advance(self, mean_reward: float, threshold: float = 0.0) -> bool:
        """Check if should advance to next stage."""
        stage = self.get_current_stage()

        # Need minimum episodes
        if self.episode_count < stage.min_episodes:
            return False

        # Need positive mean reward to advance
        if mean_reward < threshold:
            return False

        return True

    def advance_stage(self):
        """Move to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.episode_count = 0
            logger.info(f"Advanced to curriculum stage {self.current_stage}: {self.get_current_stage().name}")

    def on_episode_end(self, episode_reward: float):
        """Called at end of each episode."""
        self.episode_count += 1

    def get_initial_state(self, initial_balance: float, current_price: float) -> Tuple[float, int]:
        """
        Get initial cash and position based on current stage.

        Returns:
            (cash, position_shares)
        """
        stage = self.get_current_stage()

        if stage.start_with_position:
            position_value = initial_balance * stage.initial_position_pct
            position_shares = int(position_value / current_price)
            cash = initial_balance - (position_shares * current_price)
            return cash, position_shares
        else:
            return initial_balance, 0


# ============================================================================
# TRAINING DIAGNOSTICS
# ============================================================================

class TrainingDiagnostics:
    """
    Tracks and analyzes training metrics to identify issues:
    - Action distribution
    - Invalid action rates
    - Reward components
    - Performance metrics
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.action_counts = np.zeros(self.n_actions)
        self.invalid_action_counts = np.zeros(self.n_actions)
        self.episode_rewards = []
        self.episode_returns = []
        self.reward_components = {
            'base_return': [],
            'invalid_penalty': [],
            'trade_bonus': [],
            'risk_penalty': []
        }

    def record_action(self, action: int, was_valid: bool):
        """Record an action taken."""
        self.action_counts[action] += 1
        if not was_valid:
            self.invalid_action_counts[action] += 1

    def record_reward_components(self, components: Dict[str, float]):
        """Record individual reward components."""
        for key, value in components.items():
            if key in self.reward_components:
                self.reward_components[key].append(value)

    def record_episode(self, total_reward: float, portfolio_return: float):
        """Record episode metrics."""
        self.episode_rewards.append(total_reward)
        self.episode_returns.append(portfolio_return)

    def get_summary(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        total_actions = self.action_counts.sum()

        return {
            'action_distribution': (self.action_counts / (total_actions + 1e-8)).tolist(),
            'invalid_action_rate': (self.invalid_action_counts.sum() / (total_actions + 1e-8)),
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_portfolio_return': np.mean(self.episode_returns) if self.episode_returns else 0.0,
            'total_episodes': len(self.episode_rewards),
            'total_actions': int(total_actions)
        }

    def print_summary(self):
        """Print diagnostic summary."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print("Training Diagnostics Summary")
        print(f"{'='*60}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Total Actions: {summary['total_actions']}")
        print(f"Invalid Action Rate: {summary['invalid_action_rate']:.2%}")
        print(f"Mean Episode Reward: {summary['mean_episode_reward']:.4f}")
        print(f"Mean Portfolio Return: {summary['mean_portfolio_return']:.2%}")
        print(f"\nAction Distribution:")
        for i, pct in enumerate(summary['action_distribution']):
            print(f"  Action {i}: {pct:.2%}")
        print(f"{'='*60}\n")


# ============================================================================
# RISK MANAGEMENT (STOP-LOSS INTEGRATION)
# ============================================================================

class RiskManager:
    """
    Hard-coded risk limits that override agent actions.

    Provides protection against:
    - Large position losses (stop-loss)
    - Declining positions (trailing stop)
    - Portfolio-level drawdowns (circuit breaker)

    CRITICAL for preventing TEAM-like crashes (-32% → capped at -5%)
    """

    def __init__(
        self,
        stop_loss_pct: float = Config.RL_STOP_LOSS_PCT,
        trailing_stop_pct: float = Config.RL_TRAILING_STOP_PCT,
        max_drawdown_pct: float = Config.RL_MAX_DRAWDOWN_PCT,
        enable_stops: bool = True
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.enable_stops = enable_stops

        # Position tracking
        self.entry_price = None
        self.peak_price = None

        # Portfolio tracking
        self.peak_portfolio_value = None
        self.initial_portfolio_value = None

        logger.info(f"RiskManager initialized: stop_loss={stop_loss_pct:.1%}, "
                   f"trailing_stop={trailing_stop_pct:.1%}, max_drawdown={max_drawdown_pct:.1%}")

    def reset(self):
        """Reset risk manager state."""
        self.entry_price = None
        self.peak_price = None
        self.peak_portfolio_value = None
        self.initial_portfolio_value = None

    def on_position_entry(self, entry_price: float, portfolio_value: float):
        """
        Record entry for stop-loss tracking.

        Args:
            entry_price: Price at which position was entered
            portfolio_value: Current portfolio value
        """
        self.entry_price = entry_price
        self.peak_price = entry_price

        if self.initial_portfolio_value is None:
            self.initial_portfolio_value = portfolio_value
        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = portfolio_value

        logger.debug(f"Position entry recorded: price=${entry_price:.2f}")

    def check_stop_loss(
        self,
        current_price: float,
        portfolio_value: float,
        position: int
    ) -> Tuple[bool, str]:
        """
        Check if any stop-loss condition is triggered.

        Args:
            current_price: Current stock price
            portfolio_value: Current portfolio value
            position: Current position size (shares)

        Returns:
            (should_exit_position, reason)
        """
        if not self.enable_stops:
            return False, ""

        if position == 0:
            return False, ""

        # Update peaks
        if self.peak_price is not None:
            self.peak_price = max(self.peak_price, current_price)
        if self.peak_portfolio_value is not None:
            self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)

        # === 1. Position-Level Stop-Loss ===
        if self.entry_price is not None and current_price < self.entry_price:
            loss = (self.entry_price - current_price) / self.entry_price
            if loss > self.stop_loss_pct:
                logger.debug(f"STOP-LOSS TRIGGERED: {loss:.1%} loss from entry (${self.entry_price:.2f} → ${current_price:.2f})")
                return True, f"Stop-loss: {loss:.1%} loss"

        # === 2. Trailing Stop (from peak price) ===
        if self.peak_price is not None and current_price < self.peak_price:
            drawdown_from_peak = (self.peak_price - current_price) / self.peak_price
            if drawdown_from_peak > self.trailing_stop_pct:
                logger.debug(f"TRAILING STOP TRIGGERED: {drawdown_from_peak:.1%} from peak (${self.peak_price:.2f} → ${current_price:.2f})")
                return True, f"Trailing stop: {drawdown_from_peak:.1%} from peak ${self.peak_price:.2f}"

        # === 3. Portfolio-Level Circuit Breaker ===
        if self.peak_portfolio_value is not None and portfolio_value < self.peak_portfolio_value:
            portfolio_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if portfolio_drawdown > self.max_drawdown_pct:
                logger.debug(f"PORTFOLIO STOP TRIGGERED: {portfolio_drawdown:.1%} drawdown (${self.peak_portfolio_value:.2f} → ${portfolio_value:.2f})")
                return True, f"Portfolio stop: {portfolio_drawdown:.1%} drawdown"

        return False, ""

    def on_position_exit(self):
        """Reset position tracking when exiting."""
        self.entry_price = None
        self.peak_price = None
        logger.debug("Position exit recorded, stop tracking reset")


# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

class MarketRegime(IntEnum):
    """Market regime classification."""
    BULL = 0      # Strong uptrend (ADX > 25, upward)
    BEAR = 1      # Strong downtrend (ADX > 25, downward)
    SIDEWAYS = 2  # Choppy/ranging (ADX < 25)
    VOLATILE = 3  # High volatility regime


class RegimeDetector:
    """
    Detects market regime and provides regime-based features.

    Addresses RecurrentPPO's failure in TEAM downtrend (-5.88%).
    Adds 7 new features to observation space:
    - 4 regime one-hot features (BULL, BEAR, SIDEWAYS, VOLATILE)
    - 1 trend strength (ADX)
    - 1 trend direction (+1/-1)
    - 1 volatility regime score
    """

    def __init__(
        self,
        adx_period: int = 14,
        volatility_window: int = 20,
        high_volatility_threshold: float = 0.03  # 3% daily volatility
    ):
        self.adx_period = adx_period
        self.volatility_window = volatility_window
        self.high_volatility_threshold = high_volatility_threshold

        self.regime_history = []

    def detect_regime(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Detect current market regime.

        Args:
            prices: Historical prices (at least 50 data points recommended)
            volumes: Historical volumes (optional)

        Returns:
            (regime_id, regime_features_dict)
        """
        if len(prices) < 50:
            # Not enough data, return neutral regime
            return MarketRegime.SIDEWAYS, self._get_default_features()

        # Calculate ADX (Average Directional Index) for trend strength
        adx = self._calculate_adx(prices)

        # Calculate trend direction (20-day vs 50-day SMA)
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        trend_direction = 1.0 if sma_20 > sma_50 else -1.0

        # Calculate volatility regime
        returns = np.diff(prices[-self.volatility_window:]) / prices[-self.volatility_window:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # Classify regime
        if volatility > self.high_volatility_threshold:
            regime = MarketRegime.VOLATILE
        elif adx > 25 and trend_direction > 0:
            regime = MarketRegime.BULL
        elif adx > 25 and trend_direction < 0:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.SIDEWAYS

        # Track regime history
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)

        # Build regime features
        regime_features = {
            'regime_one_hot': np.eye(4)[regime],  # 4 features
            'trend_strength': np.array([min(adx / 100.0, 1.0)]),  # 1 feature (normalized)
            'trend_direction': np.array([trend_direction]),  # 1 feature
            'volatility_regime': np.array([min(volatility / self.high_volatility_threshold, 1.0)])  # 1 feature
        }

        logger.debug(f"Regime detected: {MarketRegime(regime).name}, ADX={adx:.1f}, "
                    f"Trend={trend_direction:.0f}, Vol={volatility:.3f}")

        return regime, regime_features

    def _get_default_features(self) -> Dict[str, np.ndarray]:
        """Return default features when not enough data."""
        return {
            'regime_one_hot': np.array([0, 0, 1, 0]),  # SIDEWAYS
            'trend_strength': np.array([0.0]),
            'trend_direction': np.array([0.0]),
            'volatility_regime': np.array([0.0])
        }

    def _calculate_adx(self, prices: np.ndarray) -> float:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength (0-100):
        - 0-25: Weak or no trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
        """
        if len(prices) < self.adx_period + 1:
            return 0.0

        # Calculate True Range components (simplified without high/low data)
        # Using price changes as proxy
        high = prices
        low = prices
        close = prices

        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth with EMA
        atr = self._ema(tr, self.adx_period)
        plus_di = 100 * self._ema(plus_dm, self.adx_period) / (atr + 1e-8)
        minus_di = 100 * self._ema(minus_dm, self.adx_period) / (atr + 1e-8)

        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = self._ema(dx, self.adx_period)

        return float(adx[-1]) if len(adx) > 0 else 0.0

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema


# ============================================================================
# MULTI-TIMEFRAME FEATURES
# ============================================================================

class MultiTimeframeFeatures:
    """
    Extracts features from multiple timeframes.

    Adds 6 new features to observation space:
    - Weekly trend (5-day SMA slope)
    - Monthly trend (20-day SMA slope)
    - Support distance (% to weekly low)
    - Resistance distance (% to weekly high)
    - Weekly price position (0-1)
    - Monthly price position (0-1)

    Total observation: 10 base + 3 trend (RecurrentPPO) + 7 regime + 6 MTF = 26 features
    """

    def __init__(
        self,
        weekly_window: int = 5,
        monthly_window: int = 20
    ):
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window

    def extract_features(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract multi-timeframe features.

        Args:
            prices: Historical prices
            volumes: Historical volumes (optional)

        Returns:
            Array of 6 features
        """
        if len(prices) < self.monthly_window:
            # Not enough data
            return np.zeros(6)

        current_price = prices[-1]

        # === Weekly Features (5 days) ===
        weekly_prices = prices[-self.weekly_window:]
        weekly_sma = np.mean(weekly_prices)
        weekly_slope = (prices[-1] - prices[-self.weekly_window]) / prices[-self.weekly_window] if len(prices) >= self.weekly_window else 0.0
        weekly_high = np.max(weekly_prices)
        weekly_low = np.min(weekly_prices)
        weekly_range = weekly_high - weekly_low
        weekly_position = (current_price - weekly_low) / (weekly_range + 1e-8)

        # === Monthly Features (20 days) ===
        monthly_prices = prices[-self.monthly_window:]
        monthly_sma = np.mean(monthly_prices)
        monthly_slope = (prices[-1] - prices[-self.monthly_window]) / prices[-self.monthly_window] if len(prices) >= self.monthly_window else 0.0
        monthly_high = np.max(monthly_prices)
        monthly_low = np.min(monthly_prices)
        monthly_range = monthly_high - monthly_low
        monthly_position = (current_price - monthly_low) / (monthly_range + 1e-8)

        # === Support/Resistance Distances ===
        support_dist = (current_price - weekly_low) / (current_price + 1e-8)
        resistance_dist = (weekly_high - current_price) / (current_price + 1e-8)

        features = np.array([
            weekly_slope,       # Trend strength (weekly)
            monthly_slope,      # Trend strength (monthly)
            support_dist,       # Distance to support (0-1)
            resistance_dist,    # Distance to resistance (0-1)
            weekly_position,    # Position in weekly range (0-1)
            monthly_position    # Position in monthly range (0-1)
        ])

        # Clip to reasonable ranges
        features = np.clip(features, -1.0, 1.0)

        return features


# ============================================================================
# KELLY CRITERION POSITION SIZING
# ============================================================================

class KellyPositionSizer:
    """
    Dynamic position sizing based on Kelly Criterion.

    Kelly Fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_win

    Adjusts BUY action sizes based on recent edge:
    - Strong edge → Larger positions
    - Weak edge → Smaller positions
    - No edge → Minimum positions
    """

    def __init__(
        self,
        max_kelly_fraction: float = 0.5,  # Use half-Kelly for safety
        min_trades_required: int = 20,     # Need history before using Kelly
        lookback_window: int = 50          # Recent trades to consider
    ):
        self.max_kelly_fraction = max_kelly_fraction
        self.min_trades_required = min_trades_required
        self.lookback_window = lookback_window

        # Trade tracking
        self.trade_results = []  # List of (is_win, pnl_pct) tuples

    def record_trade(self, entry_price: float, exit_price: float, position_size: int):
        """
        Record completed trade for Kelly calculation.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Number of shares
        """
        if position_size == 0:
            return

        pnl_pct = (exit_price - entry_price) / entry_price
        is_win = pnl_pct > 0

        self.trade_results.append((is_win, pnl_pct))

        # Keep only recent trades
        if len(self.trade_results) > self.lookback_window:
            self.trade_results.pop(0)

        logger.debug(f"Trade recorded: {'WIN' if is_win else 'LOSS'} {pnl_pct:+.2%} "
                    f"({len(self.trade_results)} trades tracked)")

    def calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly fraction based on recent performance.

        Returns:
            Kelly fraction (0.0 to max_kelly_fraction)
        """
        if len(self.trade_results) < self.min_trades_required:
            logger.debug(f"Insufficient trades for Kelly ({len(self.trade_results)}/{self.min_trades_required}), using default 0.25")
            return 0.25  # Default to 25% until enough data

        # Separate wins and losses
        wins = [pnl for is_win, pnl in self.trade_results if is_win]
        losses = [pnl for is_win, pnl in self.trade_results if not is_win]

        if len(wins) == 0 or len(losses) == 0:
            return 0.15  # Conservative if no wins or losses

        # Calculate statistics
        win_prob = len(wins) / len(self.trade_results)
        loss_prob = 1 - win_prob
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_win == 0:
            return 0.1

        # Kelly formula
        kelly = (win_prob * avg_win - loss_prob * avg_loss) / avg_win

        # Use fractional Kelly for safety (half-Kelly = max_kelly_fraction)
        safe_kelly = kelly * self.max_kelly_fraction

        # Clip to reasonable range
        safe_kelly = np.clip(safe_kelly, 0.05, 0.6)

        logger.debug(f"Kelly calculation: win_prob={win_prob:.2f}, avg_win={avg_win:+.2%}, "
                    f"avg_loss={avg_loss:.2%}, kelly={kelly:.2f}, safe_kelly={safe_kelly:.2f}")

        return float(safe_kelly)

    def adjust_action(
        self,
        base_action: int,
        use_improved_actions: bool = True
    ) -> int:
        """
        Adjust buy action based on Kelly fraction.

        Args:
            base_action: Agent's original action
            use_improved_actions: Whether using improved action space

        Returns:
            Adjusted action
        """
        # Only adjust buy actions
        if use_improved_actions:
            is_buy = base_action in [
                ImprovedTradingAction.BUY_SMALL,
                ImprovedTradingAction.BUY_MEDIUM,
                ImprovedTradingAction.BUY_LARGE
            ]
        else:
            is_buy = base_action in [TradingAction.BUY_SMALL, TradingAction.BUY_LARGE]

        if not is_buy:
            return base_action  # Don't adjust HOLD or SELL

        # Calculate Kelly fraction
        kelly_pct = self.calculate_kelly_fraction()

        # Map Kelly to actions
        if use_improved_actions:
            if kelly_pct < 0.2:
                adjusted_action = ImprovedTradingAction.BUY_SMALL
            elif kelly_pct < 0.4:
                adjusted_action = ImprovedTradingAction.BUY_MEDIUM
            else:
                adjusted_action = ImprovedTradingAction.BUY_LARGE
        else:
            if kelly_pct < 0.25:
                adjusted_action = TradingAction.BUY_SMALL
            else:
                adjusted_action = TradingAction.BUY_LARGE

        if adjusted_action != base_action:
            logger.debug(f"Kelly adjusted action: {base_action} → {adjusted_action} (kelly={kelly_pct:.2f})")

        return int(adjusted_action)
