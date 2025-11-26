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

✅ **RL Agents**: PPO, RecurrentPPO, A2C, SAC, QRDQN via Stable-Baselines3 and sb3-contrib
✅ **RecurrentPPO**: LSTM memory with trend indicators for temporal patterns
✅ **Trend Indicators**: SMA_Trend, EMA_Crossover, Price_Momentum (RecurrentPPO only)
✅ **Action Masking**: Prevents invalid trades
✅ **6-Action Space**: HOLD, BUY_SMALL/MEDIUM/LARGE, SELL_PARTIAL/ALL
✅ **Adaptive Position Sizing**: Adjusts to volatility and portfolio state
✅ **Algorithm-Specific Rewards**: Optimized per algorithm
✅ **Enhanced Rewards**: Multi-component with risk penalties, bonuses
✅ **Trading Environment**: Gymnasium-based simulation
✅ **Backtesting**: Comprehensive metrics and evaluation
✅ **Action Analysis**: Distribution visualization
✅ **Auto-Load Models**: Automatic type detection
✅ **Baseline Strategies**: Buy & Hold and Momentum
✅ **Interactive UI**: Panel-based training and backtesting
✅ **Visualization**: Progress, equity curves, actions, drawdowns

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
- `load_rl_agent()` automatically detects and loads PPO, RecurrentPPO, A2C, SAC, or QRDQN models
- `load_env_config_from_model()` loads exact training configuration from saved models
- Live trading uses loaded config to match training environment exactly

**SAC Wrapper:**
- `DiscreteToBoxWrapper` converts 6 discrete actions to continuous Box space for SAC
- Bins continuous output [-1, 1] back to discrete actions [0-5]
- Transparent wrapping with `__getattr__` forwarding

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
├── sac_discrete_wrapper.py    # SAC continuous-to-discrete action wrapper
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
  - `load_rl_agent()`: Automatically loads PPO, RecurrentPPO, A2C, SAC, or QRDQN models
  - `load_env_config_from_model()`: Loads environment config from trained model
- **sac_discrete_wrapper.py**: DiscreteToBoxWrapper for SAC
- **training.py**: `EnhancedRLTrainer` and `EnhancedTrainingConfig` using Stable-Baselines3

---

## Core Components

### 1. Trading Environment

**File**: `src/rl/environments.py`

#### Observation Space

**Shape**: `(lookback_window, num_features)`
**Default**:
- `(60, 10)` - PPO, A2C, SAC, QRDQN
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

#### A2C (Advantage Actor-Critic)

**Implementation**: Stable-Baselines3 A2C

**Strengths**:
- Native discrete action support (no wrapper needed)
- Synchronous updates with advantage estimation
- Faster training than PPO
- Simple architecture

**Hyperparameters**:
- Learning rate: `3e-4`
- n_steps: `512`
- Entropy coefficient: `0.02`
- vf_coef: `0.5`
- max_grad_norm: `0.5`
- Use RMSprop: `True`
- Normalize advantage: `True`

**Reward Config**: `A2CRewardConfig` with balanced penalties (60-70% of PPO strength)

**Note**: A2C has shown tendency toward action collapse in testing despite reward tuning attempts

#### SAC (Soft Actor-Critic)

**Implementation**: Stable-Baselines3 SAC with DiscreteToBoxWrapper

**Strengths**:
- Maximum entropy framework for exploration
- Off-policy learning with replay buffer
- Continuous action discretization

**Wrapper**:
- DiscreteToBoxWrapper converts Box[-1,1] to Discrete[0-5]
- Bins continuous actions into 6 discrete buckets

**Hyperparameters**:
- Learning rate: `3e-4`
- Buffer size: `100000`
- Learning starts: `15000`
- Train freq: `8`
- Gradient steps: `4`
- Entropy coef: `0.3`

**Reward Config**: `SACRewardConfig` with extreme reward shaping to overcome entropy bias

**Note**: SAC has shown tendency toward action collapse (100% BUY_MEDIUM) despite extreme reward shaping

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

### A2CRewardConfig

**Purpose**: Balanced penalties for synchronous actor-critic

**Key Features**:
- **Balanced Penalties**: 60-70% of PPO strength (Goldilocks zone)
- **Native Discrete Actions**: No continuous-to-discrete wrapper needed
- **Higher Entropy**: Stronger exploration (0.02 vs PPO's 0.01)
- **Longer Rollout**: n_steps=512 for better temporal context

**Configuration**:
```python
@dataclass
class A2CRewardConfig(EnhancedRewardConfig):
    risk_penalty_weight: float = 0.2       # 67% of PPO (0.3)
    drawdown_penalty_weight: float = 0.35  # 70% of PPO (0.5)
    transaction_cost_rate: float = 0.0015  # 75% of PPO (0.002)
    action_diversity_bonus: float = 0.75   # 75% of PPO (1.0)
```

**Empirical Results**:
- v1 (weak penalties): 100% BUY_MEDIUM collapse
- v2 (PPO-strength penalties): 100% SELL_PARTIAL collapse
- v3 (balanced penalties): 100% BUY_SMALL collapse

**Note**: Despite multiple tuning attempts, A2C consistently exhibits action collapse

### SACRewardConfig

**Purpose**: Extreme reward shaping to overcome SAC's entropy-seeking bias

**Key Features**:
- **Base HOLD Incentive**: +0.5 (10x stronger than original)
- **Diversity Penalty**: -1.0 for <30% diversity (prevents ALL collapse types)
- **Consecutive Penalty**: -1.0 base, scales to -5.0 max (immediate, no delay)
- **Transaction Costs**: Applied on ALL trades

**Configuration**:
```python
@dataclass
class SACRewardConfig(EnhancedRewardConfig):
    base_hold_incentive: float = 0.5
    diversity_penalty_threshold: float = 0.3
    diversity_penalty_weight: float = -1.0
    diversity_reward_threshold: float = 0.5
    diversity_reward_weight: float = 0.5
    consecutive_action_penalty: float = -1.0
    consecutive_action_max_penalty: float = -5.0
    transaction_cost_rate: float = 0.002
```

**Verified Rewards**:
- BUY spam (100%): -5.7 average reward
- HOLD spam (100%): -0.3 average reward
- Mixed actions (>50% diversity): +0.4 average reward

**Note**: Despite extreme reward shaping, SAC still collapses to 100% BUY_MEDIUM

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
    # Attempts to load: PPO → RecurrentPPO → QRDQN
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
- Algorithm selection: PPO, RecurrentPPO, A2C, SAC, QRDQN
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
    agent_type="ppo",  # or "recurrent_ppo", "a2c", "sac", "qrdqn"
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
- **Algorithms**: PPO, RecurrentPPO, A2C, SAC, QRDQN

**Training Times** (M1 Mac, 300k steps):
- PPO: 15-20 minutes
- RecurrentPPO: 25-35 minutes (LSTM requires more compute)
- A2C: 15-20 minutes
- SAC: 15-20 minutes
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

## Learning Objectives

This RL trading system is designed as an educational platform for learning modern AI/ML concepts in finance:

### Stock Analysis & Prediction
- **Technical Indicators**: Understanding RSI, MACD, Bollinger Bands, and their interpretation
- **LSTM Neural Networks**: Time series forecasting with recurrent architectures
- **Ensemble Modeling**: Combining multiple models for robust predictions
- **AI-Assisted Analysis**: Using LLMs for natural language financial insights

### Reinforcement Learning Trading
- **Policy Optimization**: Understanding PPO, actor-critic methods, and on-policy learning
- **Recurrent Policies**: LSTM-based policies for temporal pattern recognition
- **Distributional RL**: QRDQN and learning value distributions for risk-awareness
- **Trading Environment Design**: Observation spaces, action spaces, and state representation
- **Reward Engineering**: Designing reward functions to shape agent behavior
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Strategy Comparison**: Benchmarking against baselines (Buy & Hold, Momentum)
- **Action Space Design**: Discrete action spaces for trading decisions
- **Action Masking**: Preventing invalid actions in constrained environments

### Software Engineering
- **Professional UX Design**: Light theme, wide layouts, responsive design patterns
- **Real-Time Data Handling**: Caching strategies, live updates, efficient data fetching
- **Model Registry**: Automatic model discovery, versioning, and organization
- **Design Patterns**: Factory pattern, single source of truth, dataclass configuration
- **Session Persistence**: State management, auto-save, resumable sessions
- **Modular Architecture**: Clean separation of concerns, extensible design

---

## Conclusion

This RL trading system provides a complete framework for training, evaluating, and deploying RL agents for algorithmic trading. Key strengths include modular architecture, realistic simulation, comprehensive metrics, and user-friendly interface.

**Next Steps**:
1. Train your first agent (PPO, RecurrentPPO, or QRDQN recommended; A2C and SAC supported but may exhibit action collapse)
2. Backtest and compare strategies
3. Extend with custom reward functions
4. Deploy to live paper trading

---

**For questions or contributions, please refer to the project README.**
