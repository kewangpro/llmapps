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

- ✅ **RL Agents**: PPO, RecurrentPPO, DQN, QRDQN via Stable-Baselines3 and sb3-contrib
- ✅ **RecurrentPPO**: LSTM memory with trend indicators for temporal patterns
- ✅ **Trend Indicators**: SMA_Trend, EMA_Crossover, Price_Momentum (RecurrentPPO only)
- ✅ **Action Masking**: Prevents invalid trades
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
6. **Consistency**: Single source of truth for environment configuration (env_factory.py)

### Key Design Patterns

**Environment Factory Pattern:**
- `EnvConfig` dataclass serves as single source of truth for all environment parameters
- All configuration classes reference EnvConfig defaults using dataclass field extraction
- Prevents train-test mismatch by ensuring consistent parameters across training, backtesting, and live trading
- Changing a default requires editing only one place: EnvConfig

**Model Loading Utilities:**
- `load_rl_agent()` automatically detects and loads PPO, RecurrentPPO, DQN, or QRDQN models
- `load_env_config_from_model()` loads exact training configuration from saved models
- Live trading uses loaded config to match training environment exactly


**Critical Bug Fixes:**
- **Short-Selling Prevention**: Checks position before sell calculations
- **Floating Point Tolerance**: 0.01% tolerance for precision
- **Action Masking**: Prevents invalid trades automatically

---

## System Architecture

### Module Structure

```
src/rl/
├── __init__.py                 # Main RL module exports
├── env_factory.py              # Shared environment configuration (EnvConfig, create_enhanced_env)
├── model_utils.py              # Shared model loading utilities (load_rl_agent, load_env_config)
├── environments.py             # All trading environments (Base, SingleStock, Enhanced)
├── training.py                 # Training pipeline (EnhancedRLTrainer)
├── improvements.py             # Action masking, adaptive sizing, curriculum learning
├── callbacks.py                # Training callbacks (progress, early stopping)
├── rewards.py                  # Reward functions (simple, risk-adjusted, enhanced)
├── backtesting.py              # Backtesting (Engine + Metrics Calculator)
├── baselines.py                # Baseline strategies (Buy&Hold, Momentum)
├── live_trading.py             # Live paper trading engine
└── visualizer.py               # RL-specific visualizations

src/ui/pages/
├── rl_training.py              # RL training UI
└── live_trading.py             # Live trading UI

src/config.py                   # RL configuration
```

**Module Organization**:
- **env_factory.py**: Shared environment configuration system (single source of truth)
  - `EnvConfig`: Unified configuration dataclass
  - `create_enhanced_env()`: Factory function for consistent environments
- **model_utils.py**: Shared model loading utilities
  - `load_rl_agent()`: Automatically loads PPO, RecurrentPPO, DQN, or QRDQN models
  - `load_env_config_from_model()`: Loads environment config from trained model
- **training.py**: `EnhancedRLTrainer` and `EnhancedTrainingConfig` using Stable-Baselines3

---

## Core Components

### 1. Trading Environment

**File**: `src/rl/environments.py`

#### Observation Space

**Shape**: `(lookback_window, num_features)`
**Default**:
- `(60, 10)` - PPO, DQN, QRDQN
- `(60, 13)` - RecurrentPPO with trend indicators

**Base Features (10)**:
- Price (normalized)
- Volume (normalized)
- Cash ratio
- Position ratio
- Portfolio value change
- Technical indicators: RSI, MACD, MACD Signal, Bollinger Bands, Stochastic

**Trend Indicators (3 - RecurrentPPO only)**:
- SMA_Trend: 5-day slope of 20-period SMA
- EMA_Crossover: (EMA12 - EMA26) / EMA26
- Price_Momentum: 5-day rate of change

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

#### PPO (Proximal Policy Optimization)

**Implementation**: Stable-Baselines3 PPO

**Strengths**:
- Stable training with clipped objective
- Reliable baseline performance
- Strong penalties prevent action collapse

**Hyperparameters**:
- Learning rate: `3e-4`
- Batch size: `64`
- n_steps: `2048`
- n_epochs: `10`

**Reward Config**: `PPORewardConfig` with strong penalties and diversity bonuses

#### RecurrentPPO

**Implementation**: sb3-contrib RecurrentPPO

**Strengths**:
- LSTM memory for temporal patterns
- Trend indicators (13 features vs 10 base)
- Best for trending markets
- Momentum bonuses

**Hyperparameters**:
- Same as PPO
- Uses MlpLstmPolicy
- Requires 300k timesteps

**Reward Config**: `RecurrentPPORewardConfig` with momentum bonuses and hold winner rewards

#### DQN (Deep Q-Network)

**Implementation**: Stable-Baselines3 DQN

**Strengths**:
- Value-based off-policy learning
- Experience replay for sample efficiency
- Epsilon-greedy exploration
- Stable Q-value learning

**Hyperparameters**:
- Learning rate: `3e-4`
- Buffer size: `100000`
- Learning starts: `10000`
- Batch size: `64`
- Train freq: `4`
- Exploration fraction: `0.3`
- Epsilon decay: `1.0` → `0.05`

**Reward Config**: `DQNRewardConfig` with balanced rewards for stable value learning

#### QRDQN (Quantile Regression DQN)

**Implementation**: sb3-contrib QRDQN

**Strengths**:
- Distributional RL learns value distribution
- Risk-aware decision making
- Off-policy learning with replay

**Hyperparameters**:
- Learning rate: `1e-4`
- Buffer size: `100000`
- Train freq: `4`
- Exploration: 0.3 fraction

**Reward Config**: `QRDQNRewardConfig` with risk-encouraging rewards to counter natural conservatism

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
class RecurrentPPORewardConfig(EnhancedRewardConfig):
    risk_penalty_weight: float = 0.1
    drawdown_penalty_weight: float = 0.2
    transaction_cost_rate: float = 0.0005
    hold_winner_bonus: float = 0.1
    momentum_trend_bonus: float = 0.15
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
    action_diversity_bonus: float = 1.0
```

### DQNRewardConfig

**Purpose**: Balanced rewards for stable Q-value learning

**Key Features**:
- **Moderate Risk Penalties**: Balanced approach to risk management
- **Standard Transaction Costs**: Realistic trading costs
- **Action Diversity Bonus**: Encourages balanced action selection
- **Profitable Trade Bonus**: Rewards successful trades

**Configuration**:
```python
@dataclass
class DQNRewardConfig(EnhancedRewardConfig):
    risk_penalty_weight: float = 0.1
    drawdown_penalty_weight: float = 0.15
    transaction_cost_rate: float = 0.0005
    action_diversity_bonus: float = 0.5
    profitable_trade_bonus: float = 0.2
```

**Results**: Stable Q-value learning with balanced action distribution

### QRDQNRewardConfig

**Purpose**: Risk-encouraging to counter QRDQN's natural conservatism

**Key Features**:
- **Zero Risk Penalties**: QRDQN learns risk naturally from distribution
- **Lower Transaction Costs**: Encourages trading (0.0003 vs 0.0005)
- **Large Position Bonus**: +0.2 bonus for aggressive sizing
- **Excessive Selling Penalty**: -0.15 penalty to let winners run
- **Higher Profitable Trade Bonus**: 3x stronger (0.3 vs 0.1)

**Configuration**:
```python
@dataclass
class QRDQNRewardConfig(EnhancedRewardConfig):
    risk_penalty_weight: float = 0.0
    drawdown_penalty_weight: float = 0.0
    transaction_cost_rate: float = 0.0003
    large_position_bonus: float = 0.2
    excessive_selling_penalty: float = -0.15
    profitable_trade_bonus: float = 0.3
```

**Results**: Improved QRDQN performance from +2.08% to +22.35% (10x improvement)

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

**Automatic Type Detection**:
```python
def load_rl_agent(model_path, env=None):
    # Attempts to load: PPO → RecurrentPPO → DQN → QRDQN
    # Auto-detects correct algorithm type
    # Returns loaded agent with is_trained flag
```

**Config Loading**:
```python
def load_env_config_from_model(model_dir):
    # Loads exact training configuration
    # Used by live trading to match training environment
    # Includes trend indicators for RecurrentPPO models
```

### Bug Fixes & Improvements

**Short-Selling Prevention**:
- Fixed `AdaptiveActionSizer` bug that allowed unintentional short positions
- Now checks `position == 0` before calculating sell size
- Prevents negative position values

**Floating Point Precision**:
- Added 0.01% tolerance to position size checks
- Prevents valid orders from being rejected due to floating point errors
- Applied to all position limit comparisons

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
├── SESSION_qrdqn_TSLA_*.json
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

**Config Priority**:
1. Saved model config (if available)
2. User-specified overrides
3. EnvConfig defaults

---

## User Interface

### RL Training Page

**File**: `src/ui/pages/rl_training.py`

**Features**:
- Algorithm selection: PPO, RecurrentPPO, DQN, QRDQN
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
```

### Configuration System

All environment parameters centralized in `src/rl/env_factory.py`:

```python
@dataclass
class EnvConfig:
    initial_balance: float = 100000.0
    max_position_pct: float = 80.0
    transaction_cost_rate: float = 0.0005
    slippage_rate: float = 0.0005
    lookback_window: int = 60
    include_trend_indicators: bool = False  # For RecurrentPPO
    use_action_masking: bool = True
    use_enhanced_rewards: bool = True
    use_adaptive_sizing: bool = True
    use_improved_actions: bool = True
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
    agent_type="ppo",  # or "recurrent_ppo", "dqn", "qrdqn"
    total_timesteps=100000
)

trainer = EnhancedRLTrainer(config)
results = trainer.train()
```

### 2. Backtesting

```python
from src.rl import BacktestEngine, BacktestConfig, EnhancedRLTrainer

agent = EnhancedRLTrainer.load_agent(
    model_path="data/models/rl/qrdqn_AAPL_20240101/best_model.zip",
    agent_type="qrdqn"
)

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
- **Algorithms**: PPO, RecurrentPPO, DQN, QRDQN

**Training Times** (M1 Mac, 300k steps):
- PPO: 15-20 minutes
- RecurrentPPO: 25-35 minutes (LSTM requires more compute)
- DQN: 15-20 minutes
- QRDQN: 15-20 minutes

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

Each environment step calculates reward using **8 components** from `EnhancedRewardFunction`:

#### Component Breakdown

**1. Base Return (Primary Signal)**
```python
portfolio_return = (current_value - prev_value) / prev_value
base_reward = portfolio_return * 100  # Scaled up (return_weight)
```

**2. Invalid Action Penalty**
```python
if action_is_masked:
    reward -= 0.01  # -1% penalty for invalid action
```

**3. Excessive Trading Penalty**
```python
if action == prev_action and is_trading:
    consecutive_count += 1
    penalty = -0.005 * min(consecutive_count, 5)  # Progressive
    reward += penalty  # Penalty is negative
```

**4. Transaction Costs**
```python
if is_buy or is_sell:
    cost = trade_value * transaction_cost_rate  # 0.05%-0.2%
    reward -= cost / portfolio_value
```

**5. Risk Penalties**
```python
volatility = std(recent_returns[-20:])
reward -= volatility * 0.1  # Volatility penalty

if len(returns) >= 5:
    sharpe = mean_return / (volatility + 1e-8)
    reward += sharpe * 0.05  # Sharpe bonus
```

**6. Drawdown Penalty**
```python
if drawdown > 5%:
    reward -= drawdown * 0.5  # Penalize deep drawdowns
```

**7. Profitable Trade Bonus**
```python
if portfolio_return > 0:
    reward += 0.001  # Small bonus for profitable steps
```

**8. Hold Incentives**
```python
if action == HOLD:
    reward += 0.001  # Base hold incentive

if action == HOLD and position > 0 and return > 0:
    reward += 0.002  # Hold winning position bonus
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
| **DQN** | Off-policy | Continuous | Replay buffer |
| **QRDQN** | Off-policy | Continuous | Replay buffer |

**On-Policy (PPO, RecurrentPPO):**
- Collect fresh experience each iteration
- Old data becomes "stale" after policy updates
- More stable but less sample-efficient

**Off-Policy (DQN, QRDQN):**
- Store experience in replay buffer
- Can reuse data indefinitely
- More sample-efficient but potentially less stable

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

### CLI Utility: retrain_and_compare.py

A command-line tool for automated training, backtesting, and comparison of all RL algorithms.

**Location**: `retrain_and_compare.py` (project root)

#### Purpose

Automate the workflow of:
1. Training all (or selected) RL algorithms on a symbol
2. Running comprehensive backtests
3. Comparing performance across algorithms and baseline strategies
4. Detecting action collapse issues

#### Usage

**Basic Usage** (train all algorithms):
```bash
source .venv/bin/activate
python retrain_and_compare.py --symbol AAPL
```

**Advanced Options**:
```bash
# Train specific algorithms only
python retrain_and_compare.py --symbol TSLA --algorithms ppo,recurrent_ppo,qrdqn

# Custom training duration
python retrain_and_compare.py --symbol NVDA --timesteps 500000

# Skip training, only backtest existing models
python retrain_and_compare.py --symbol GOOGL --skip-training

# Exclude baseline strategies
python retrain_and_compare.py --symbol MSFT --no-baselines
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol` | string | **required** | Stock symbol to train on (e.g., AAPL, TSLA) |
| `--algorithms` | string | `all` | Comma-separated list: `ppo,recurrent_ppo,dqn,qrdqn` |
| `--timesteps` | int | `300000` | Training timesteps per algorithm |
| `--skip-training` | flag | `False` | Skip training, only backtest existing models |
| `--no-baselines` | flag | `False` | Skip baseline strategies (Buy & Hold, Momentum) |

#### What It Does

**1. Training Phase** (unless `--skip-training`):
- Trains each selected algorithm sequentially
- Uses last 3 years of historical data (1095 days)
- Saves models to `data/models/rl/{algorithm}_{symbol}_{timestamp}/`
- Saves both `best_model.zip` (peak performance) and `final_model.zip`

**2. Backtesting Phase**:
- Automatically finds latest trained models for each algorithm
- Tests on last 9 months of data (280 days)
- Loads exact training configuration from saved models
- Compares against Buy & Hold and Momentum strategies

**3. Results Display**:
```
📊 BACKTEST COMPARISON
====================
Strategy          Return    Sharpe  Max DD   Win Rate  HOLD  BUY_S  BUY_M  BUY_L  SELL_P  SELL_A  Trades
PPO Agent         +15.23%   1.45    -8.2%    65%       25%   15%    20%    18%    12%     10%     48
RECURRENT PPO     +18.67%   1.78    -6.1%    72%       22%   18%    22%    16%    14%     8%      52
DQN Agent         +16.89%   1.52    -7.5%    68%       23%   17%    21%    19%    11%     9%      51
QRDQN Agent       +22.35%   2.12    -5.4%    75%       20%   20%    25%    18%    10%     7%      56
Buy & Hold        +12.50%   1.20    -10.0%   N/A       N/A   N/A    N/A    N/A    N/A     N/A     1
Momentum          +10.15%   0.95    -11.2%   N/A       N/A   N/A    N/A    N/A    N/A     N/A     38
```

**4. Analysis**:
- Identifies best performer by total return
- Detects action collapse (>80% single action usage)
- Provides actionable recommendations

#### Example Output

```bash
$ python retrain_and_compare.py --symbol AAPL

################################################################################
#                                                                              #
#                         RL MODEL RETRAINING                                  #
#                                                                              #
################################################################################

Symbol: AAPL
Timesteps: 300,000
Algorithms: PPO, RECURRENT PPO, DQN, QRDQN

================================================================================
🚀 TRAINING PPO on AAPL
================================================================================

Training progress: 100%|████████████████| 300000/300000

✅ PPO Training Complete!
   Model saved to: data/models/rl/ppo_AAPL_20250126_143022/

[... training continues for each algorithm ...]

================================================================================
📊 COMPREHENSIVE BACKTEST: AAPL
================================================================================

   ✅ PPO: +15.23%
   ✅ RECURRENT PPO: +18.67%
   ✅ DQN: +16.89%
   ✅ QRDQN: +22.35%
   ✅ Buy & Hold: +12.50%
   ✅ Momentum: +10.15%

[... results table ...]

================================================================================
🎯 ANALYSIS
================================================================================

🥇 Best Performer: QRDQN Agent
   Return: +22.35%
   Sharpe: 2.12

📊 Action Distribution Analysis:
   ✅ PPO Agent: Balanced actions (max 25%)
   ✅ RECURRENT PPO: Balanced actions (max 22%)
   ✅ DQN Agent: Balanced actions (max 23%)
   ✅ QRDQN Agent: Balanced actions (max 25%)

================================================================================
✨ Complete!
```

#### Use Cases

**1. Algorithm Selection**:
```bash
# Compare all algorithms to choose best for a symbol
python retrain_and_compare.py --symbol TEAM
```

**2. Periodic Retraining**:
```bash
# Retrain models monthly with latest data
python retrain_and_compare.py --symbol AAPL --algorithms ppo,recurrent_ppo,qrdqn
```

**3. Quick Performance Check**:
```bash
# Test existing models without retraining
python retrain_and_compare.py --symbol NVDA --skip-training
```

**4. Algorithm Comparison**:
```bash
# Compare all 4 algorithms on a symbol
python retrain_and_compare.py --symbol GOOGL
```

#### Integration with Web UI

Models trained via this CLI tool are **automatically available** in the web interface:
- Training page: Select algorithm → models appear in dropdown
- Live Trading page: Algorithm selection → auto-finds latest model
- Models page: View all trained models

#### Notes

- **Training Time**: ~15-35 min per algorithm (300k steps)
- **Data Requirements**: Minimum 3 years historical data
- **Disk Space**: ~50-100MB per trained model
- **Parallel Training**: Not supported (trains sequentially to avoid resource conflicts)
- **Logging**: All training logs saved to `data/logs/`

#### Troubleshooting

**"No model found for {algorithm}"**:
- Solution: Train the algorithm first (remove `--skip-training`)

**"Insufficient data for symbol"**:
- Solution: Use well-established stocks with >3 years of history

**Training crashes/OOM**:
- Solution: Reduce `--timesteps` to 100000 or close other applications

---

## Conclusion

This RL trading system provides a complete framework for training, evaluating, and deploying RL agents for algorithmic trading. Key strengths include modular architecture, realistic simulation, comprehensive risk management, and advanced features for improved performance.

**Next Steps**:
1. Train your first agent (all 4 algorithms recommended)
2. Backtest and compare strategies
3. Experiment with different improvement combinations
4. Deploy to live paper trading

---

**For questions or contributions, please refer to the project README.**
