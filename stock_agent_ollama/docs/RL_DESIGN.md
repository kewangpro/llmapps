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

✅ **RL Agents**: PPO, RecurrentPPO, QRDQN via Stable-Baselines3 and sb3-contrib
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
- `load_rl_agent()` automatically detects and loads PPO, RecurrentPPO, SAC, or QRDQN models
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
  - `load_rl_agent()`: Automatically loads PPO, RecurrentPPO, SAC, or QRDQN models
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
- `(60, 10)` - PPO, SAC, QRDQN
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
    agent_type="a2c",  # or "ppo", "recurrent_ppo", "sac", "qrdqn"
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
- **Algorithms**: PPO, RecurrentPPO, SAC, QRDQN

**Training Times** (M1 Mac, 300k steps):
- PPO: 15-20 minutes
- RecurrentPPO: 25-35 minutes (LSTM requires more compute)
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

## Conclusion

This RL trading system provides a complete framework for training, evaluating, and deploying RL agents for algorithmic trading. Key strengths include modular architecture, realistic simulation, comprehensive metrics, and user-friendly interface.

**Next Steps**:
1. Train your first agent (PPO, RecurrentPPO, A2C recommended, SAC not recommended, or QRDQN)
2. Backtest and compare strategies
3. Extend with custom reward functions
4. Deploy to live paper trading

---

**For questions or contributions, please refer to the project README.**
