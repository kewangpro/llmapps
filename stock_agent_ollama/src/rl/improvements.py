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

from .environments import TradingAction, BaseTradingEnv

logger = logging.getLogger(__name__)


# ============================================================================
# IMPROVED ACTION SPACE
# ============================================================================

class ImprovedTradingAction(IntEnum):
    """Improved discrete trading actions with HOLD as default (action 0)."""
    HOLD = 0          # Default action - do nothing
    BUY_SMALL = 1     # Buy with 10-20% of cash (adaptive)
    BUY_MEDIUM = 2    # Buy with 20-40% of cash (adaptive)
    BUY_LARGE = 3     # Buy with 40-60% of cash (adaptive)
    SELL_PARTIAL = 4  # Sell 50% of position
    SELL_ALL = 5      # Sell entire position


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
        max_position_size: int,
        portfolio_value: float,
        max_position_pct: float = 40.0
    ) -> np.ndarray:
        """
        Get binary mask of valid actions (1=valid, 0=invalid).

        Args:
            cash: Available cash
            position: Current position size (shares)
            current_price: Current stock price
            max_position_size: Maximum shares allowed
            portfolio_value: Total portfolio value
            max_position_pct: Max position as % of portfolio value

        Returns:
            Binary mask array
        """
        mask = np.ones(self.n_actions, dtype=np.float32)

        if self.use_improved_actions:
            return self._get_improved_action_mask(
                cash, position, current_price, max_position_size,
                portfolio_value, max_position_pct
            )
        else:
            return self._get_standard_action_mask(
                cash, position, current_price, max_position_size,
                portfolio_value, max_position_pct
            )

    def _get_standard_action_mask(
        self, cash: float, position: int, current_price: float,
        max_position_size: int, portfolio_value: float, max_position_pct: float
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
            # Check position limit
            shares_small = int(buy_small_amount / current_price)
            if position + shares_small > max_position_size:
                mask[TradingAction.BUY_SMALL] = 0.0
            # Check percentage limit
            new_position_value = (position + shares_small) * current_price
            if (new_position_value / portfolio_value * 100) > max_position_pct:
                mask[TradingAction.BUY_SMALL] = 0.0

        # BUY_LARGE (30% of cash)
        buy_large_amount = cash * 0.3
        if buy_large_amount < min_buy_cost:
            mask[TradingAction.BUY_LARGE] = 0.0
        else:
            shares_large = int(buy_large_amount / current_price)
            if position + shares_large > max_position_size:
                mask[TradingAction.BUY_LARGE] = 0.0
            new_position_value = (position + shares_large) * current_price
            if (new_position_value / portfolio_value * 100) > max_position_pct:
                mask[TradingAction.BUY_LARGE] = 0.0

        # Ensure at least HOLD is always valid
        mask[TradingAction.HOLD] = 1.0

        return mask

    def _get_improved_action_mask(
        self, cash: float, position: int, current_price: float,
        max_position_size: int, portfolio_value: float, max_position_pct: float
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
                if position + shares > max_position_size:
                    mask[action] = 0.0
                new_position_value = (position + shares) * current_price
                if (new_position_value / portfolio_value * 100) > max_position_pct:
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
    """Configuration for enhanced reward function."""
    # Base rewards
    return_weight: float = 1.0

    # Penalties
    invalid_action_penalty: float = -0.5
    excessive_trading_penalty: float = -0.05
    risk_penalty_weight: float = 0.3
    drawdown_penalty_weight: float = 0.5

    # Bonuses
    profitable_trade_bonus: float = 0.1
    sharpe_bonus_weight: float = 0.2
    valid_action_bonus: float = 0.01

    # Transaction costs
    transaction_cost_rate: float = 0.001
    slippage_rate: float = 0.0005

    # Shaping parameters
    use_action_shaping: bool = True
    use_progress_shaping: bool = True
    use_risk_shaping: bool = True

    # Advanced
    min_hold_steps: int = 5  # Minimum steps to hold before selling


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

    def calculate(
        self,
        portfolio_value: float,
        action: int,
        prev_action: int,
        cash: float,
        position: float,
        price: float,
        prev_price: float,
        max_position_size: int = 1000,
        max_position_pct: float = 40.0,
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
            max_position_size: Max shares allowed
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
                max_position_size=max_position_size,
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
            if action == self.prev_action and action != 0:  # Not HOLD
                self.consecutive_same_actions += 1
                if self.consecutive_same_actions > 3:
                    reward += self.config.excessive_trading_penalty
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
        if action != prev_action and action != 0:  # Action changed and not HOLD
            transaction_value = price * max(position, 100)  # Estimate
            transaction_cost = transaction_value * self.config.transaction_cost_rate
            slippage = transaction_value * self.config.slippage_rate
            total_cost = (transaction_cost + slippage) / self.prev_portfolio_value
            reward -= total_cost

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
            Number of shares to sell
        """
        if use_improved_actions:
            if action == ImprovedTradingAction.SELL_PARTIAL:
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
