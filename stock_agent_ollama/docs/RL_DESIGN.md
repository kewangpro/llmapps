# Reinforcement Learning Trading System - Design & Implementation

**Version:** 2.0.0
**Last Updated:** 2025

---

## Overview

### Purpose

This RL trading system provides advanced reinforcement learning capabilities for the Stock Agent platform, enabling users to:

- **Train RL agents** to learn trading strategies through environment interaction
- **Backtest strategies** with comprehensive risk metrics
- **Compare performance** against baseline strategies
- **Visualize results** through interactive dashboards

### Key Features

- ✅ **RL Agents**: PPO, RecurrentPPO, Ensemble via Stable-Baselines3 and sb3-contrib
- ✅ **RecurrentPPO**: LSTM memory with trend indicators for temporal patterns
- ✅ **Trend Indicators**: SMA_Trend, EMA_Crossover, Price_Momentum (RecurrentPPO only)
- ✅ **Action Masking**: Prevents invalid trades automatically during execution
- ✅ **6-Action Space**: HOLD, BUY_SMALL/MEDIUM/LARGE, SELL_PARTIAL/ALL
- ✅ **Adaptive Position Sizing**: Adjusts to volatility and portfolio state
- ✅ **Algorithm-Specific Rewards**: Optimized per algorithm
- ✅ **Enhanced Rewards**: Multi-component with risk penalties, bonuses
- ✅ **Advanced Risk Management**: Stop-loss, trailing stops, circuit breakers
- ✅ **Market Regime Detection**: BULL, BEAR, SIDEWAYS, VOLATILE
- ✅ **Multi-Timeframe Features**: Weekly/monthly trend analysis
- ✅ **Kelly Position Sizing**: Optimal sizing based on edge
- ✅ **Ensemble Agents**: Combine multiple algorithms
- ✅ **Trading Environment**: Gymnasium-based simulation
- ✅ **Backtesting**: Comprehensive metrics and evaluation
- ✅ **Action Analysis**: Distribution visualization
- ✅ **Auto-Load Models**: Automatic type detection
- ✅ **Baseline Strategies**: Buy & Hold and Momentum
- ✅ **Interactive UI**: Panel-based training and backtesting
- ✅ **Visualization**: Progress, equity curves, actions, drawdowns

### Design Philosophy

1. **Modularity**: Clean separation between environments, training, and backtesting
2. **Extensibility**: Easy to add new reward functions or strategies
3. **Realism**: Configurable transaction costs and slippage
4. **Educational**: Transparent metrics and visualizations for learning
5. **Integration**: Seamlessly extends existing stock analysis platform
6. **Consistency**: Single source of truth for configuration (src/config.py)
7. **Type Safety**: Type-safe enums (src/rl/types.py) prevent magic number bugs
8. **Testability**: Comprehensive test suite (48 tests, 16% coverage)

### Key Design Patterns

**Environment Factory Pattern:**
- `EnvConfig` dataclass serves as single source of truth for all environment parameters
- All configuration classes reference EnvConfig defaults using dataclass field extraction
- Prevents train-test mismatch by ensuring consistent parameters across training, backtesting, and live trading
- Changing a default requires editing only one place: EnvConfig

**Model Loading Utilities:**
- `load_rl_agent()` automatically detects and loads PPO, RecurrentPPO, or Ensemble models
- `load_env_config_from_model()` loads exact training configuration from saved models
- Live trading uses loaded config to match training environment exactly

**Configuration Management:**
- Centralized in `src/config.py` with Config.RL_* constants
- Eliminates redundant _ENV_DEFAULTS across modules
- Single source of truth for all RL parameters
- Environment variables support for easy overrides

**Type Safety:**
- `src/rl/types.py` provides TradingAction and ImprovedTradingAction enums
- Eliminates magic numbers in code (e.g., action == 2 becomes action == TradingAction.BUY_SMALL)
- Ensemble agent uses type-safe conflict resolution (Buy vs Sell → HOLD)

**Critical Bug Fixes:**
- **Short-Selling Prevention**: Checks position before sell calculations
- **Floating Point Tolerance**: 0.01% tolerance for precision
- **Action Masking**: Prevents invalid trades automatically
- **Data Freshness**: Fixed missing last data point in backtesting and charts by adjusting environment step limits and inclusive date ranges
- **Data Model Restoration**: Fixed missing Portfolio, Position, Trade, Order classes

---

## System Architecture

### Module Structure

```
src/rl/
├── __init__.py                 # Main RL module exports
├── types.py                    # Type-safe action enums (TradingAction, ImprovedTradingAction)
├── env_factory.py              # Shared environment configuration (EnvConfig, create_enhanced_env)
├── model_utils.py              # Shared model loading utilities (load_rl_agent, load_env_config)
├── environments.py             # All trading environments (Base, SingleStock, Enhanced)
├── training.py                 # Training pipeline (EnhancedRLTrainer)
├── improvements.py             # Action masking, adaptive sizing, curriculum learning
├── ensemble.py                 # Ensemble agent with type-safe conflict resolution
├── callbacks.py                # Training callbacks (progress, early stopping)
├── rewards.py                  # Reward functions (simple, risk-adjusted, enhanced)
├── backtesting.py              # Backtesting (Engine + Metrics Calculator)
├── baselines.py                # Baseline strategies (Buy&Hold, Momentum)
├── live_trading.py             # Live paper trading engine (with data models)
└── visualizer.py               # RL-specific visualizations

src/ui/pages/
├── rl_training.py              # RL training UI
└── live_trading.py             # Live trading UI

src/config.py                   # Centralized RL configuration (Config.RL_*)

tests/                          # Test suite (48 tests)
├── test_config.py              # Configuration tests
├── test_action_masking.py      # Action masking validation
├── test_live_trading_models.py # Data model tests
├── test_rl_components.py       # RL environment tests
└── test_technical_analysis.py  # Indicator tests

retrain_rl.py          # Training automation CLI (root directory)
validate_backtest.py            # Backtest validation CLI (root directory)
```

**Module Organization**:
- **env_factory.py**: Shared environment configuration system (single source of truth)
  - `EnvConfig`: Unified configuration dataclass
  - `create_enhanced_env()`: Factory function for consistent environments
- **model_utils.py**: Shared model loading utilities
  - `load_rl_agent()`: Automatically loads PPO, RecurrentPPO, or Ensemble models
  - `load_env_config_from_model()`: Loads environment config from trained model
- **training.py**: `EnhancedRLTrainer` and `EnhancedTrainingConfig` using Stable-Baselines3

---

## Core Components

### 1. Trading Environment

**File**: `src/rl/environments.py`

#### Observation Space

**Shape**: `(lookback_window, num_features)`
**Default**:
- `(60, 10)` - PPO, Ensemble (base + technical)
- `(60, 13)` - RecurrentPPO with trend indicators
- `(60, 26)` - EnhancedTradingEnv with improved actions + regime + MTF

**Base Features (5)**:
- Price (normalized)
- Volume (normalized)
- Cash ratio
- Position ratio
- Portfolio value change

**Technical Indicators (5)**:
- RSI, MACD, MACD Signal, Bollinger Bands, Stochastic

**Trend Indicators (3 - RecurrentPPO only)**:
- SMA_Trend: 5-day slope of 20-period SMA
- EMA_Crossover: (EMA12 - EMA26) / EMA26
- Price_Momentum: 5-day rate of change

**Invalid Action Handling**:
- Action masking prevents invalid actions (e.g., BUY with no cash) from being executed
- If an agent predicts an invalid action, it is automatically replaced with HOLD
- Action masks are NOT included in the observation space to avoid feature bloat
- Agents learn valid behavior through the reward signal from successful trades

#### Action Space

**Type**: Discrete (6 actions)

| Action | Value | Description |
|--------|-------|-------------|
| HOLD | 0 | No action (default, safe) |
| BUY_SMALL | 1 | Buy with ~15% of cash |
| BUY_MEDIUM | 2 | Buy with ~30% of cash |
| BUY_LARGE | 3 | Buy with ~50% of cash |
| SELL_PARTIAL | 4 | Sell 50% of position |
| SELL_ALL | 5 | Sell entire position |

---

### 2. RL Agents

**PPO (Proximal Policy Optimization)**

**Implementation**: Stable-Baselines3 PPO

**Strengths**:
- Stable training with clipped objective
- Exceptional risk-adjusted returns (2.59 Sharpe ratio on PLTR)
- Strong baseline performance with balanced action distribution
- No VecNormalize normalization for better convergence

**Hyperparameters**:
- Learning rate: `3e-4`
- Batch size: `128`
- n_steps: `2048`
- n_epochs: `10`
- Entropy coefficient: `0.2`

**Reward Config**: `PPORewardConfig` with strong penalties and diversity bonuses

#### RecurrentPPO

**Implementation**: sb3-contrib RecurrentPPO

**Strengths**:
- LSTM memory for temporal patterns
- Trend indicators (13 features vs 10 base)
- Good performance for trending markets (0.10 Sharpe ratio on PLTR)
- Stable performance with momentum-focused rewards
- No VecNormalize normalization preserves temporal learning

**Hyperparameters**:
- Learning rate: `3e-4`
- Batch size: `128`
- n_steps: `2048`
- n_epochs: `10`
- Entropy coefficient: `0.2`
- Policy: `MlpLstmPolicy`
- Requires 300k timesteps

**Reward Config**: `RecurrentPPORewardConfig` with momentum bonuses and hold winner rewards

#### Ensemble (PPO + RecurrentPPO)

**Implementation**: Custom `EnsemblePPOAgent` combining both policy gradient methods

**Strategy**:
- PPO (30% weight): Opportunistic growth for tactical trades
- RecurrentPPO (70% weight): Primary strategy with LSTM memory and trend-following
- *Weights are configurable via `ensemble_config.json`*

**Decision Logic**:
1. Get predictions from both PPO and RecurrentPPO
2. If agents agree → Use that action
3. If agents disagree → Weighted vote with confidence-based scoring
4. Confidence weighting: Uses action probabilities to determine final decision

**Strengths**:
- Robust risk-adjusted returns (0.94 Sharpe ratio on PLTR)
- Leverages RecurrentPPO's LSTM memory as primary strategy (70% weight)
- PPO provides opportunistic growth for tactical trades (30% weight)
- Confidence-based decisions favor more certain predictions
- Balanced action distribution with healthy win rate (48%)

**Training**:
- Trains both PPO and RecurrentPPO independently
- Combines trained models with optimized 30/70 weighting
- Saves both component models plus ensemble metadata
- **Artifacts**: Generates standalone `ppo_best_model.zip` and `recurrent_ppo_best_model.zip` which can be loaded and used independently of the ensemble wrapper

**Observation Space Handling**:
- PPO expects base features (e.g., base + technical + regime + MTF)
- RecurrentPPO expects base features + trend indicators
- Ensemble intelligently routes observations:
  - When receiving observations with trend indicators: strips them for PPO, keeps them for RecurrentPPO
  - When receiving base observations: passes same observation to both models
- This allows heterogeneous observation spaces within a single ensemble

**Reward Config**: Uses `PPORewardConfig` for PPO component and `RecurrentPPORewardConfig` for RecurrentPPO component

---

## Deep Dive: Understanding PPO

**PPO (Proximal Policy Optimization)** is the industry standard for Reinforcement Learning because it strikes a perfect balance between **ease of implementation**, **sample efficiency**, and **performance**.

### 1. The Core Concept: "Learning Safely"
In traditional Reinforcement Learning (like Policy Gradients), an agent tweaks its neural network weights to increase the probability of actions that resulted in high rewards.

- **The Problem**: Sometimes, an agent will experience a "lucky" run (e.g., buying a stock right before a random spike). In older algorithms, the agent might drastically rewrite its weights to *always* do that action. If that action was just luck, the agent's performance collapses (catastrophic forgetting).
- **The PPO Solution**: PPO allows the agent to update its policy but puts a **limit (a "clip")** on how much the policy can change in a single training step. It effectively says: *"Improve the policy, but don't change the probability of an action by more than 20% (0.2) at a time."* This "Proximal" constraint ensures **stability**.

### 2. How it Works (Actor-Critic)
PPO is an **Actor-Critic** algorithm, meaning it uses two neural networks (or two "heads") that work together:

1. **The Actor (The Trader)**:
   - **Job**: Observes the market state (prices, RSI, MACD) and outputs **Action Probabilities** (e.g., 10% HOLD, 80% BUY_LARGE, 10% SELL).
   - **Goal**: Maximize the reward.
2. **The Critic (The Analyst)**:
   - **Job**: Observes the market state and predicts the **Value ($V$)** of being in that state (i.e., "How much profit do I expect to make from this point forward?").
   - **Goal**: Accurately predict the final outcome.

The **Critic** helps the **Actor** learn by reducing variance. The Actor looks at how much better the outcome was *compared to what the Critic expected* (the **Advantage**).

### 3. Why PPO is Great for Stock Trading
Financial data is extremely **noisy** and **non-stationary** (market rules change over time).

- **Stability**: Because PPO limits update size, it is less likely to overfit to short-term trends (like a 2-week bull run) and forget how to survive a crash.
- **On-Policy Learning**: PPO learns from recent data and discards it. This ensures it optimizes based on its *current* strategy, rather than outdated history.
- **Robustness**: PPO works well with default hyperparameters, avoiding weeks of math tuning.

### 4. PPO vs. RecurrentPPO
- **Standard PPO**: Looks at the fixed window (e.g., 60 days) as a static snapshot.
- **RecurrentPPO**: Adds an **LSTM (Long Short-Term Memory)** layer. It remembers the *sequence* of events, allowing it to understand context like "we have been in a downtrend for 3 months, but momentum is shifting."

In this project:
- **PPO** provides the reliable, low-risk foundation.
- **RecurrentPPO** adds memory for complex trend capture.
- **Ensemble** combines them for robust performance.

---

## Algorithm-Specific Reward Configurations

### RecurrentPPORewardConfig

**Purpose**: Optimized for trend-following with LSTM memory

**Key Features**:
- **Momentum Bonuses**: Rewards staying long during strong uptrends
- **Hold Winner Bonus**: Rewards holding profitable positions during trends
- **Reduced Penalties**: Lower risk penalties to allow trend-following behavior
- **13-Feature Observation**: 10 base + 3 trend indicators (SMA_Trend, EMA_Crossover, Price_Momentum)

**Configuration**:
```python
@dataclass
class RecurrentPPORewardConfig(PPORewardConfig):
    # Balanced penalties for trend following
    risk_penalty_weight: float = 0.1
    drawdown_penalty_weight: float = 0.2

    # Higher incentives for holding winning positions
    hold_winning_position_bonus: float = 0.2
    momentum_trend_bonus: float = 0.3

    # Penalize high-frequency trading
    excessive_trading_penalty: float = -0.2

    # Profitable trade bonus (kept moderate)
    profitable_trade_bonus: float = 0.05

    # Minimal transaction costs
    transaction_cost_rate: float = 0.001
```

### PPORewardConfig

**Purpose**: Strong penalties to prevent action collapse

**Key Features**:
- **Strong Risk Penalties**: High weights to discourage excessive risk
- **Diversity Bonuses**: Encourages using multiple actions
- **Action Collapse Prevention**: Empirically proven to prevent single-action convergence

**Configuration**:
```python
@dataclass
class PPORewardConfig(EnhancedRewardConfig):
    risk_penalty_weight: float = 0.3
    drawdown_penalty_weight: float = 0.5
    transaction_cost_rate: float = 0.002
    diversity_bonus: float = 0.5
    diversity_penalty: float = -0.5
```

---

## Architecture & Design Patterns

### Environment Factory Pattern

**File**: `src/rl/env_factory.py`

**Purpose**: Single source of truth for all environment configuration

**Key Components**:
- **EnvConfig Dataclass**: Centralized configuration with defaults
- **create_enhanced_env()**: Factory function ensures consistency
- **Prevents Train-Test Mismatch**: Same config across training/backtesting/live trading
- **Conditional Features**: Automatically enables trend indicators for RecurrentPPO

**Benefits**:
- Change defaults in one place
- No parameter duplication
- Guaranteed consistency
- Type safety with dataclasses

### Model Loading & Detection

**File**: `src/rl/model_utils.py`

**Path Normalization** (Single Source of Truth):
```python
def normalize_model_path(model_path, agent_type=None):
    # Converts any path format to correct loading format
    # Handles: directories, files, ensemble subdirs, legacy formats
    # Auto-detects agent type from training_config.json
    # Returns: Path to model file (PPO/RecurrentPPO) or directory (Ensemble)
```

**Automatic Type Detection**:
```python
def load_rl_agent(model_path, env=None):
    # Attempts to load: PPO → RecurrentPPO → Ensemble
    # Auto-detects correct algorithm type from training config
    # Supports backward compatibility with all legacy path formats
    # Returns loaded agent with is_trained flag
```

**Config Loading**:
```python
def load_env_config_from_model(model_dir):
    # Loads exact training configuration
    # Used by live trading to match training environment
    # Includes trend indicators for RecurrentPPO models
```

**Backward Compatibility**:
- Supports legacy directory paths (converted to file paths automatically)
- Handles ensemble subdirectory structures (both new and old formats)
- Auto-detects model type when agent_type not specified
- Seamless session resumption across all saved formats

### Bug Fixes & Improvements

**Short-Selling Prevention**:
- Fixed `AdaptiveActionSizer` bug that allowed unintentional short positions
- Now checks `position == 0` before calculating sell size
- Prevents negative position values

**Floating Point Precision**:
- Added 0.01% tolerance to position size checks
- Prevents valid orders from being rejected due to floating point errors
- Applied to all position limit comparisons

**Data Freshness & Last Point Visibility**:
- Fixed issue where backtesting and charts were missing the most recent 1-2 days of data
- Updated `max_steps` calculation in `EnhancedTradingEnv` to include the final row of historical data
- Synchronized date range logic between data fetching and visualization to ensure consistency
- Backtesting now correctly processes the full range up to the current date

**Symbol Input Flexibility**:
- Removed restrictions on predefined ticker symbols
- Now accepts any valid ticker from Yahoo Finance
- Improved user experience

**Position Limits Consistency**:
- Corrected default max position from 40% to 80% across all systems
- Applied to training, backtesting, and live trading
- Ensures consistent behavior

---

## Live Trading Implementation

### Multi-Session Support

**File**: `src/rl/live_trading.py`

**Features**:
- Run multiple trading strategies simultaneously
- Independent session state and persistence
- Session-specific configuration and models

**Session Management**:
```python
data/live_sessions/
├── SESSION_ppo_AAPL_*.json
├── SESSION_ensemble_TSLA_*.json
└── SESSION_recurrent_ppo_NVDA_*.json
```

### Session Persistence

**Auto-Save Triggers**:
- Every 5 minutes during active trading
- On session stop/pause
- On application shutdown
- On position changes

**Persisted Data**:
- Portfolio state (cash, positions, total value)
- Trade history with timestamps
- Agent configuration and model path
- Risk management settings
- Session metadata (start time, symbol, algorithm)

**Auto-Resume**:
- Automatically loads sessions on app restart
- Restores exact portfolio state
- Continues from last checkpoint
- No data loss

### Environment Matching

**Training Config Loading**:
- Live trading loads exact training configuration from saved models
- Matches `lookback_window`, `transaction_cost_rate`, `slippage_rate`
- Automatically enables trend indicators for RecurrentPPO models
- Ensures no train-test distribution mismatch

**Churn Protection**:
- Prevents rapid trade reversals within 15 minutes
- Requires significant price movement (>1%) to override cooldown
- Reduces transaction cost erosion from "noise" trading
- Logs protection events for transparency

**Config Priority**:
1. Saved model config (if available)
2. User-specified overrides
3. EnvConfig defaults

---

## User Interface

### RL Training Page

**File**: `src/ui/pages/rl_training.py`

**Features**:
- Algorithm selection: PPO, RecurrentPPO, Ensemble
- Symbol input (accepts any valid ticker)
- Training period configuration (default: 1095 days / 3 years)
- Training steps (default: 300k)
- Real-time progress tracking
- Training diagnostics display
- Action distribution visualization

**Training Results Display**:
- Summary card with agent details
- Training diagnostics (invalid action rate, episode reward, portfolio return)
- Metrics (win rate, explained variance)
- Progress chart showing reward improvement
- Action distribution pie chart

**Backtest Results Display**:
- Performance comparison table with all strategies
- Portfolio value comparison chart
- Stock price with trade signals chart (visualizes buy/sell actions on price)
- Action distribution comparison chart
- Key metrics comparison chart

### Live Trading Page

**File**: `src/ui/pages/live_trading.py`

**Features**:
- Session creation and management
- Real-time portfolio tracking
- Position monitoring with unrealized P&L
- Trade history with agent decisions
- Event log for system notifications
- Pause/Resume/Stop controls
- Risk management settings

**UI Enhancements**:
- Wider model name column (improved readability)
- Algorithm buttons: 350px width (+50px from original)
- "Allow Extended Hours" checkbox on separate row
- Better session management interface
- Optimized granular updates for high-performance rendering

### Design System

**Theme**: Light theme with professional aesthetics
- White/light gray color scheme
- High contrast for readability
- Color-coded values (green=positive, red=negative)
- Clean, modern design

**Layout Philosophy**:
- Wide horizontal layouts for widescreen monitors
- Minimal vertical scrolling
- Dashboard: Markets + Quick Actions side-by-side
- Analysis: 70/30 split (Chart + Signals/Predictions)
- Optimized for desktop and laptop screens

**Live Data Updates**:
- Real-time market indices (5-second refresh)
- Live watchlist prices with color-coded changes
- Interactive charts with hover details
- Auto-loading of trained models

---

## Implementation Details

### Dependencies

```txt
# Reinforcement Learning
gymnasium>=0.29.0
stable-baselines3>=2.2.0
sb3-contrib>=2.2.0

# Portfolio Optimization
PyPortfolioOpt>=1.5.0
cvxpy>=1.4.0

# Data Layer
yfinance>=0.2.28              # Yahoo Finance API
```

### Data Infrastructure

The RL system leverages an intelligent multi-tier caching system for optimal performance:

**Cache Tiers:**
- **Real-time quotes**: 1 minute TTL for live trading and backtesting
- **Intraday data**: 15 minutes TTL for 1m/5m interval data
- **Bulk data**: 5 minutes TTL for Top Movers scanning in auto-select mode
- **Company fundamentals**: 1 hour TTL for stable company information
- **Historical OHLCV**: 1 day TTL for training and backtesting data

This caching strategy balances data freshness with API rate limits and performance, ensuring training and live trading have access to current market data without unnecessary API calls.

### Configuration System

All environment parameters centralized in `src/rl/env_factory.py`:

```python
@dataclass
class EnvConfig:
    initial_balance: float = 100000.0
    max_position_pct: float = 80.0
    transaction_cost_rate: float = 0.0  # $0 commissions
    slippage_rate: float = 0.0005  # 0.05% slippage
    lookback_window: int = 60
    include_trend_indicators: bool = False  # For RecurrentPPO
    use_action_masking: bool = True
    use_enhanced_rewards: bool = True
    use_adaptive_sizing: bool = True
    use_improved_actions: bool = True
    use_vec_normalize: bool = False  # Disabled - improves convergence

# Note: VecNormalize is disabled by default as it was found to hurt performance
# by ~75% for both PPO and RecurrentPPO. The non-stationary normalization
# statistics interfere with policy learning.
```

---

## Usage Guide

### 1. Training an Agent

```python
from src.rl import EnhancedRLTrainer, EnhancedTrainingConfig

config = EnhancedTrainingConfig(
    symbol="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
    agent_type="ppo",  # or "recurrent_ppo", "ensemble"
    total_timesteps=300000
)

trainer = EnhancedRLTrainer(config)
results = trainer.train()
```

### 2. Backtesting

```python
from src.rl import BacktestEngine, BacktestConfig, EnhancedRLTrainer

# Load trained agent
agent = load_rl_agent("data/models/rl/ppo_AAPL_20240101/best_model.zip")

config = BacktestConfig(
    symbol="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-01"
)

engine = BacktestEngine(config)
result = engine.run_agent_backtest(agent)

print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
```

---

## Performance Metrics

### Training Performance

**Optimal Configuration**:
- **Training Period**: 1095 days (3 years)
- **Training Steps**: 300,000 (recommended for all algorithms)
- **Algorithms**: PPO, RecurrentPPO, Ensemble

**Training Times** (M1 Mac, 300k steps):
- PPO: 15-20 minutes
- RecurrentPPO: 25-35 minutes (LSTM requires more compute)
- Ensemble: 40-55 minutes (trains both PPO + RecurrentPPO)

**Model Saving**:
- `best_model.zip`: Peak mean reward (used for backtesting)
- `final_model.zip`: End of training

---

## Extension Points

### Adding New Algorithms

To add a new algorithm, update `src/rl/training.py`:

```python
elif agent_type_lower == 'new_algo':
    from stable_baselines3 import NewAlgo
    agent_class = NewAlgo
    policy_type = 'MlpPolicy'
```

Then update `src/rl/model_utils.py` for loading.

---

## Training Mechanics: Iterations and Rewards

### What is an Iteration?

In PPO and RecurrentPPO (on-policy algorithms), an **iteration** is one complete learning cycle consisting of two phases:

#### 1. Rollout Phase (Experience Collection)
- Agent interacts with environment for `n_steps` (default: **2048 steps**)
- Each step: agent observes → selects action → receives reward
- Experience stored in rollout buffer: states, actions, rewards, values

#### 2. Training Phase (Policy Update)
- Sample mini-batches from the 2048 collected steps
- Update policy network using those batches
- Repeat for `n_epochs` (default: **10 epochs**)

**Key Training Parameters:**
```python
n_steps = 2048       # Steps to collect before training
batch_size = 128     # Samples processed together
n_epochs = 10        # Training passes over collected data
```

**Understanding Training Logs:**
```
total_timesteps | 2048   ← 2048 steps collected
iterations      | 1      ← Completed iteration #1
ep_rew_mean     | 19.4   ← Average episode reward
ep_len_mean     | 691    ← Average episode length
```

**Iteration Contains Multiple Episodes:**
```
Iteration 1 (2048 steps):
├─ Episode 1: 691 steps → total_reward = 18.98
├─ Episode 2: 691 steps → total_reward = 19.37
└─ Episode 3: 666 steps → total_reward = 18.50 (partial)
                         ─────────────────────────
                         Total: 2048 steps collected → Train policy
```

### How Rewards are Calculated Per Step

Each environment step calculates reward using **7 components** from `EnhancedRewardFunction`:

#### Component Breakdown

**1. Base Return (Primary Signal)**
```python
portfolio_return = (current_value - prev_value) / prev_value
base_reward = portfolio_return * 1.0  # (return_weight)
```

**2. Action Masking**
```python
if action_is_masked:
    action = HOLD  # Replace invalid action with HOLD
    reward += invalid_action_penalty # -1.0
else:
    reward += valid_action_bonus # 0.01
```

**3. Excessive Trading Penalty**
```python
if is_trading_action and was_trading_action:
    consecutive_count += 1
    penalty = excessive_trading_penalty * min(consecutive_count, 5)  # Progressive
    if action == prev_action:
        penalty *= 2.0 # Double penalty for hyper-churn
    reward += penalty
```

**4. Transaction Costs**
```python
if is_buy or is_sell:
    cost = trade_value * transaction_cost_rate
    reward -= cost / portfolio_value
```

**5. Risk Penalties**
```python
volatility = std(recent_returns[-20:])
reward -= volatility * risk_penalty_weight

if len(returns) >= 5:
    sharpe = mean_return / (volatility + 1e-8)
    reward += sharpe * sharpe_bonus_weight
```

**6. Drawdown Penalty**
```python
if drawdown > 5%:
    reward -= drawdown * drawdown_penalty_weight
```

**7. Profitable Trade Bonus**
```python
if portfolio_return > 0:
    reward += profitable_trade_bonus # 0.01-0.05
```

**8. Hold Incentives**
```python
if action == HOLD:
    reward += base_hold_incentive # 0.01-0.05

if action == HOLD and position > 0 and return > 0:
    reward += hold_winning_position_bonus # 0.05-0.2
```

#### Real Example Calculation

**Scenario:** Bought stock yesterday, holding today, price up 0.5%

```python
# Step 1: Portfolio increased
portfolio_return = 0.005  # +0.5%
reward = 0.005 * 100 = 0.5  # Base reward

# Step 2: Action = HOLD (valid, no trading)
reward += 0.001  # Valid action bonus
reward += 0.001  # Base hold incentive
reward += 0.002  # Hold winning position bonus

# Step 3: No transaction costs (HOLD doesn't trade)
# cost = 0

# Step 4: Low volatility
volatility_penalty = -0.0001

# Final Reward
reward = 0.5 + 0.004 - 0.0001 ≈ 0.504
```

### Episode vs Iteration Rewards

**Episode Reward:**
- Sum of all step rewards in one episode
- Episode = one complete trading period (e.g., 691 steps ≈ 3 years)

```python
episode_reward = sum(all_step_rewards)
# Example: 691 steps × 0.028 avg reward/step ≈ 19.4
```

**Mean Reward (in Training Logs):**
```python
mean_reward = average(last_100_episodes)
# From log: "ep_rew_mean: 19.4"
```

**Interpreting Mean Reward of 19.4:**
- Average reward per step: 19.4 / 691 ≈ **0.028**
- Since reward ≈ portfolio_return × 100
- Average daily return ≈ **0.028%**
- Annualized return ≈ **7%** (252 trading days)

This indicates the agent is learning profitable, low-risk trading strategies with proper risk management.

### Algorithm Differences

| Algorithm | Learning Type | Steps/Update | Data Reuse |
|-----------|--------------|--------------|------------|
| **PPO** | On-policy | 2048 | 10 epochs |
| **RecurrentPPO** | On-policy | 2048 | 10 epochs |
| **Ensemble** | On-policy | 2048 | 10 epochs (both components) |

**All Algorithms (PPO, RecurrentPPO, Ensemble):**
- Use on-policy learning with policy gradients
- Collect fresh experience each iteration
- Old data becomes "stale" after policy updates
- More stable and proven effective for trading

---

## Advanced Improvements

The system includes comprehensive risk management and feature enhancements to improve trading performance and safety:

### Priority Improvements Implementation

**Status: All Implemented and Integrated**

#### 1. Stop-Loss Integration

**File**: `src/rl/improvements.py:1011-1128`
**Class**: `RiskManager`

Prevents catastrophic losses through automated risk controls:

**Features**:
- **5% stop-loss** per position (from entry price)
- **3% trailing stop** (from peak price)
- **15% portfolio circuit breaker** (from peak portfolio value)
- Automatically forces SELL_ALL when triggered

**Integration**:
- Added to `EnhancedTradingEnv` with `use_risk_manager=True`
- Checked in `step()` before action execution
- Resets on episode reset

**Impact**: Caps losses at -5% per trade, preventing large drawdowns

---

#### 2. Market Regime Detection

**File**: `src/rl/improvements.py:1134-1287`
**Class**: `RegimeDetector`

Helps agents adapt to different market conditions:

**Features**:
- Detects 4 market regimes: BULL, BEAR, SIDEWAYS, VOLATILE
- Uses ADX (Average Directional Index) for trend strength
- Adds **7 new features** to observation space:
  - 4 one-hot regime indicators
  - 1 trend strength (ADX)
  - 1 trend direction (+1/-1)
  - 1 volatility regime

**Integration**:
- Added to `EnhancedTradingEnv` with `use_regime_detector=True`
- Features computed in `_get_observation()`
- Observation space expanded by 7 features

**Impact**: Improves performance in varying market conditions

---

#### 3. Ensemble Agent

**File**: `src/rl/ensemble.py`
**Classes**: `EnsembleAgent`, `AdaptiveEnsembleAgent`

Combines strengths of multiple algorithms:

**Features**:
- Weighted voting across multiple trained models
- Confidence scoring (Herfindahl-Hirschman Index)
- Support for both recurrent and non-recurrent agents
- Optional adaptive weights based on recent performance

**Usage Example**:
```python
from src.rl.ensemble import EnsembleAgent
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

# Load trained models
ppo = PPO.load("ppo_model.zip")
rppo = RecurrentPPO.load("rppo_model.zip")

# Create ensemble
ensemble = EnsembleAgent([
    (ppo, 0.35),   # Weight by validation Sharpe
    (rppo, 0.45),  # Best in uptrends
])

# Predict
action, confidence = ensemble.predict_with_confidence(obs)
```

**Impact**: Combines algorithm strengths for more robust trading

---

#### 4. Multi-Timeframe Features

**File**: `src/rl/improvements.py:1293-1372`
**Class**: `MultiTimeframeFeatures`

Improves trend identification with multiple timeframes:

**Features**:
- Adds **6 new features** to observation space:
  - Weekly trend slope (5-day SMA)
  - Monthly trend slope (20-day SMA)
  - Support distance (% to weekly low)
  - Resistance distance (% to weekly high)
  - Weekly price position (0-1)
  - Monthly price position (0-1)

**Integration**:
- Added to `EnhancedTradingEnv` with `use_mtf_features=True`
- Features computed in `_get_observation()`
- Observation space expanded by 6 features

**Impact**: Better trend identification across timeframes

---

#### 5. Kelly Position Sizing

**File**: `src/rl/improvements.py:1378-1517`
**Class**: `KellyPositionSizer`

Optimizes position sizes based on win rate and edge:

**Features**:
- Kelly Criterion: `f = (p*W - q*L) / W`
  - p = win probability
  - W = average win
  - q = loss probability (1-p)
  - L = average loss
- Half-Kelly for safety (max 50%)
- Requires minimum 20 trades before activating
- Adjusts BUY action sizes: BUY_SMALL → BUY_MEDIUM → BUY_LARGE

**Integration**:
- Added to `EnhancedTradingEnv` with `use_kelly_sizing=True`
- Adjusts actions in `step()` before execution
- Records trade results for edge calculation

**Impact**: Optimal position sizing based on statistical edge

---

### Enhanced Observation Space

The observation space expands based on enabled features:

| Feature Group | Count | Enabled By |
|--------------|-------|------------|
| Base features | 5 | Always |
| Technical indicators | 5 | `include_technical_indicators=True` (default) |
| Trend indicators | 3 | `include_trend_indicators=True` (RecurrentPPO) |
| Regime features | 7 | `use_regime_detector=True` |
| Multi-timeframe | 6 | `use_mtf_features=True` |
| **Total** | **26** | All enabled |

**Observation shape**: `(lookback_window, num_features)` = `(60, 26)` when all features enabled

---

### Usage Example

All improvements integrate into `EnhancedTradingEnv`:

```python
from src.rl.environments import EnhancedTradingEnv

env = EnhancedTradingEnv(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2023-12-31",
    # Risk management
    use_risk_manager=True,
    stop_loss_pct=0.05,           # 5% stop-loss
    trailing_stop_pct=0.03,        # 3% trailing stop
    max_drawdown_pct=0.15,         # 15% circuit breaker
    # Enhanced features
    use_regime_detector=True,      # Market regime detection
    use_mtf_features=True,         # Multi-timeframe features
    use_kelly_sizing=True,         # Kelly position sizing
    # Existing improvements
    use_action_masking=True,
    use_enhanced_rewards=True,
    use_adaptive_sizing=True,
    use_improved_actions=True
)
```

---

## Developer Tools

### Training Automation Tool

**File**: `retrain_rl.py` (project root)

#### Purpose

Automates the end-to-end training workflow for RL agents. It handles data fetching, environment setup, model training, and automatically triggers an initial validation backtest to generate performance artifacts.

#### Usage

**Basic Usage** (train all algorithms on a single stock):
```bash
source .venv/bin/activate
python retrain_rl.py --symbol AAPL
```

**Advanced Options**:
```bash
# Train on multiple stocks
python retrain_rl.py --symbol AMZN,AAPL,META --algorithms ensemble

# Train specific algorithms only
python retrain_rl.py --symbol TSLA --algorithms ppo,recurrent_ppo

# Custom training duration
python retrain_rl.py --symbol NVDA --timesteps 500000
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol` | string | **required** | Stock symbol(s) to train on - single or comma-separated (e.g., AAPL or AAPL,TSLA,NVDA) |
| `--algorithms` | string | `all` | Comma-separated list: `ppo,recurrent_ppo,ensemble`<br>Controls which algorithms to train |
| `--timesteps` | int | `300000` | Training timesteps per algorithm |

#### Workflow

**1. Training Phase**:
- Trains each selected algorithm sequentially
- Uses last 3 years of historical data (1095 days)
- Saves models to `data/models/rl/{algorithm}_{symbol}_{timestamp}/`
- Saves both `best_model.zip` (peak performance) and `final_model.zip`

**2. Artifact Generation**:
- Automatically invokes `validate_backtest.py` logic upon completion
- Generates `backtest_results.json` immediately
- Ensures models are ready for evaluation tools (`eval_training.py`) without manual steps

**3. Next Steps**:
- After training, use `eval_training.py` to compare performance and detect pathologies.
- Use `validate_backtest.py` for detailed mathematical checks.

---

### Backtest Validation Tool

**File**: `validate_backtest.py` (project root)

#### Purpose

Provides comprehensive validation of backtest results to ensure mathematical correctness and detect common bugs such as data leakage, calculation errors, or unrealistic performance metrics.

#### Usage

**Basic Usage**:
```bash
source .venv/bin/activate

# Validate ALL models for a symbol (all algorithms, all versions)
python validate_backtest.py --symbol AAPL

# Force a fresh backtest run before validation
python validate_backtest.py --symbol AAPL --run

# Validate all watchlist stocks (all algorithms, all models)
python validate_backtest.py --watchlist
```

**Advanced Options**:
```bash
# Validate only LATEST model (faster for quick checks)
python validate_backtest.py --symbol TEAM --algorithm ppo --latest-only

# Force run specific algorithm across watchlist
python validate_backtest.py --watchlist --algorithm ensemble --run
```

#### Command-Line Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--symbol` | string | N/A | N/A | Stock symbol to validate (mutually exclusive with `--watchlist`) |
| `--watchlist` | flag | N/A | N/A | Validate all symbols from default watchlist |
| `--algorithm` | string | `all` | `ppo`, `recurrent_ppo`, `ensemble`, `all` | Algorithm(s) to validate |
| `--latest-only` | flag | `False` | N/A | Validate only the most recent model (default: validates all models) |
| `--run` | flag | `False` | N/A | Force execution of a new backtest before validation |

**Note**: By default, the tool validates **ALL available models** for the selected algorithm(s), not just the latest. This enables comprehensive validation across model versions. Results show full model names with timestamps (e.g., `ppo_TEAM_20260106_071323`).

#### Validation Checks

The tool performs **10 comprehensive checks**:

1. **Return Calculation**: Verifies (final_value - initial_value) / initial_value matches reported return
2. **Action Distribution**: Ensures action percentages sum to 100%
3. **Win Rate Calculation**: Validates winning_trades / completed_trades matches reported win rate
4. **Portfolio Value Consistency**: Checks first value = initial capital, last value = final value, no negative values
5. **Metrics Reasonableness**:
   - Sharpe ratio < 8.0 (warns if > 6.0 as "excellent")
   - Win rate thresholds tiered by sample size (<20: 95%, 20-49: 93%, 50-99: 91%, 100+: 90%)
6. **Individual Trade Validation**: Validates P&L calculations on paired round-trip trades (entry/exit price, commission)
7. **Transaction Cost Inclusion**: Verifies trades include non-zero `cost` or `commission` fields (default 0.05% per trade)
8. **Return Reconciliation from Trades**: Validates portfolio value = cash + position value, and cash matches net trade flows
9. **Market Data Integrity**: Verifies backtest trade prices match actual historical market data
10. **Reproducibility Test**: Automated double-run verification for RL agents to detect non-deterministic behavior

#### Example Output

```bash
$ python validate_backtest.py --symbol META

╔════════════════════════════════════════════════════════════════════╗
║                     BACKTEST VALIDATION REPORT                     ║
║                             META - PPO                             ║
╚════════════════════════════════════════════════════════════════════╝

======================================================================
                     Check 1: Return Calculation
======================================================================

  Initial Portfolio Value: $100,000.00
  Final Portfolio Value:   $110,243.74
  Calculated Return:       10.24%
  Reported Return:         10.24%
  Difference:              0.0000%
✅ PASS Return calculation

[... additional checks ...]

======================================================================
                  Check 9: Reproducibility Test
======================================================================

  Running automated verification for PPO...
  (Executing second backtest pass to verify determinism)

  Comparison:
    Pass 1: +10.2437% return, 0 trades
    Pass 2: +1.5934% return, 57 trades
❌ FAIL Reproducibility
      Return mismatch: +10.2437 vs +1.5934; Trade count mismatch: 0 vs 57
⚠️  WARNING: Strategy is non-deterministic! Evaluation results may be unreliable.

======================================================================
                          VALIDATION SUMMARY
======================================================================

  Checks Passed: 8/9
  Success Rate:  88.9%

⚠️  MOSTLY PASSED - Review warnings above
```

#### Batch Validation Output

When validating multiple combinations (e.g., `--watchlist --algorithm all`), the tool provides an overall summary table:

```
============================================================================================================================================
                                                         OVERALL VALIDATION SUMMARY                                                         
============================================================================================================================================

Symbol     Model                                         Return     Passed   Status     Details/Warnings
--------------------------------------------------------------------------------------------------------------------------------------------
META       ppo_META_20260107_024443                      +10.24%    8/9      ⚠️  WARN Reproducibility
META       recurrent_ppo_META_20260107_031932            -3.55%     8/9      ⚠️  WARN Reproducibility
META       ensemble_META_20260107_024443                 -3.87%     8/9      ⚠️  WARN Reproducibility
META       Buy & Hold Baseline                           -5.14%     9/9      ✅ PASS 
META       Momentum Baseline                             -8.94%     9/9      ✅ PASS 

Overall: 42/45 checks passed (93.3%)
⚠️  3 validation(s) need attention
============================================================================================================================================
```

#### Transaction Cost Handling

The validation tool checks for transaction costs in trade records. The backtesting system applies:

- **Commissions**: $0 (zero-commission era: Fidelity, Schwab, Robinhood)
- **Slippage rate**: 0.05% (liquid S&P 500 stocks)
- **Total cost per trade**: 0.05% of trade value (0.1% round-trip)

Trade records include transaction costs:
- Backtesting: Uses `cost` field with total slippage
- Live trading: Uses `commission` field with total slippage
- Validator: Automatically checks for either field

**Configuration:**
```python
# In env_factory.py (EnvConfig)
transaction_cost_rate: float = 0.0      # $0 commissions
slippage_rate: float = 0.0005           # 0.05% slippage
```

#### Integration with Training/Backtesting

The validation tool automatically:
- Loads backtest results from `data/models/rl/{algo}_{symbol}_{timestamp}/backtest_results.json`
- Finds the most recent model for each symbol/algorithm combination
- Validates enhanced backtest data including:
  - Portfolio value history
  - Action distribution
  - Individual trades with costs
  - Initial/final portfolio values

#### Notes

- **Data Requirements**: Requires backtest results with enhanced validation data (portfolio_values, action_distribution, trades)
- **Backtest Format**: Works with backtest results from models trained after validation enhancements were added
- **Sample Size Awareness**: Win rate thresholds adjust based on completed trades to avoid false positives
- **Educational Use**: Helps users understand and trust backtest metrics

---

### Training Evaluation Tool

**File**: `eval_training.py` (project root)

#### Purpose

Scans all trained RL models to analyze performance patterns, detect training pathologies (like action collapse), and provide high-level insights into the overall health of the agent ecosystem.

#### Usage

**Basic Usage**:
```bash
# Evaluate all models
python eval_training.py

# Filter by symbol
python eval_training.py --symbol PLTR
```

**Advanced Options**:
```bash
# Filter by minimum trades (e.g., ignore inactive models)
python eval_training.py --min-trades 10

# Sort results by different metrics
python eval_training.py --sort return    # Sort by total return
python eval_training.py --sort sharpe    # Sort by Sharpe ratio
python eval_training.py --sort winrate   # Sort by win rate
python eval_training.py --sort maxdd     # Sort by max drawdown (lowest risk first)
python eval_training.py --sort age       # Sort by most recently validated (default)

# Prune (archive) models with negative returns older than 24 hours
python eval_training.py --prune --min-return 0 --age 24h
```

#### Key Features

- **Pathology Detection**:
  - **Action Collapse**: Warns if >80% of actions are identical (e.g., stuck on HOLD).
  - **Over-trading**: Warns if trades per day > 2.0.
  - **Under-trading**: Identifies agents that rarely trade.
- **Pruning System**: Automatically moves models matching specific criteria (e.g., negative returns and older than specified age) to a `data/models/archive/` folder to reduce noise.
- **Performance Grading**: Color-coded status (Green/Yellow/Red) based on returns and Sharpe ratio.
- **Strategic Insights**: Aggregates data to find the best performing algorithms and symbols.
- **Visual Report**: Generates a readable table with key metrics (Return, Sharpe, MaxDD, Win Rate) and detailed insights.

#### Integration

This tool complements `retrain_rl.py` (generation) and `validate_backtest.py` (validation) by providing the **evaluation and monitoring** layer of the RL pipeline.

---

## Conclusion

This RL trading system provides a complete framework for training, evaluating, and deploying RL agents for algorithmic trading. Key strengths include modular architecture, realistic simulation, comprehensive risk management, and advanced features for improved performance.

**Next Steps**:
1. Train your first agent (all 3 algorithms recommended)
2. Backtest and compare strategies
3. Experiment with different improvement combinations
4. Deploy to live paper trading

---

**For questions or contributions, please refer to the project README.**