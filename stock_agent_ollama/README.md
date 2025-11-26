# 🤖 Stock Agent Pro

A professional financial analysis platform combining **AI-powered analysis**, **LSTM neural networks**, and **reinforcement learning** for comprehensive stock analysis, predictions, and automated trading strategies.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![AI](https://img.shields.io/badge/AI-Ollama%20%2B%20LSTM%20%2B%20RL-orange)
![Educational](https://img.shields.io/badge/Purpose-Educational-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🎯 What It Does

### 📊 Professional Dashboard
- **Market Overview** with live major indices (S&P 500, NASDAQ, Dow Jones, Russell 2000)
- **Interactive Watchlist** in sidebar with clickable stock cards
  - Real-time prices and daily changes
  - Live position tracking from active trading sessions
  - Click any stock to instantly navigate to Analysis page
  - Automatic updates every 5 seconds
- **Quick Actions** for common tasks (Train LSTM, Backtest, Live Trading, Report)
- **Light Theme** professional interface optimized for wide screens

### 📈 Stock Analysis & Prediction
- **Interactive Charts** with candlestick patterns and volume
- **Technical Indicators** (RSI, MACD, Bollinger Bands, Moving Averages)
- **30-Day LSTM Predictions** using ensemble neural networks (3 models)
- **AI-Powered Analysis** with natural language insights
- **Trading Signals** (BUY/SELL/HOLD) with confidence scores

![Stock Analysis Screenshot](docs/screenshots/analysis.png)
*Stock analysis with interactive charts, technical indicators, and AI-powered insights*

### 🤖 Reinforcement Learning Trading
- **Train RL Agents** using PPO, RecurrentPPO, SAC, and QRDQN with action masking
- **6-Action Trading Space** (HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_PARTIAL, SELL_ALL)
- **RecurrentPPO** with LSTM memory for temporal pattern recognition in trending markets
- **Trend Indicators** for RecurrentPPO (SMA_Trend, EMA_Crossover, Price_Momentum)
- **Adaptive Position Sizing** that adjusts to market volatility
- **Training Metrics** (Win Rate, Action Distribution, Episode Rewards, Explained Variance)
- **Algorithm-Specific Rewards** with optimized configs per algorithm
- **Comprehensive Backtesting** with automated best model loading
- **Strategy Comparison** against Buy & Hold and Momentum baselines
- **Performance Metrics** (Returns, Sharpe Ratio, Max Drawdown, Win Rate)
- **Visualization Charts** (Performance comparison, Action distribution, Key metrics)

![RL Training Screenshot](docs/screenshots/training.png)
*Train and backtest RL agents with comprehensive performance metrics and strategy comparison*

### 🔴 Live Trading Simulation
- **Paper Trading** with real-time market data (Yahoo Finance)
- **Trained Agent Execution** using PPO, RecurrentPPO, SAC, or QRDQN models in live markets
- **Persistent Sessions** that automatically save and resume across restarts
  - Portfolio state preserved
  - Trade history maintained
  - Configuration retained
- **Real-time Portfolio Tracking** with live P&L updates
- **Risk Management** (stop-loss, position limits, circuit breakers)
- **Live Monitoring** with trading status, positions, and event log
- **Session Management** via `session_manager.py`
- **Educational Platform** for safe strategy testing with virtual capital

![Live Trading Screenshot](docs/screenshots/live_trade.png)
*Real-time paper trading with persistent sessions, portfolio tracking, and risk management*

### 🗂️ Model Registry
- **LSTM Models** with performance metrics (Final Loss, Val Loss)
- **RL Agents** with training dates and algorithm types
- **Model Management** with automatic discovery and organization

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Ollama (Optional but Recommended)
```bash
# Install from https://ollama.ai
ollama pull gemma3:latest
```

### 3. Launch Platform
```bash
python src/main.py
# Open http://localhost:5006
```

### 4. Start Using

**Quick Market Check:**
- Open Dashboard → View market indices and watchlist

**Stock Analysis:**
- Click Analysis → Select symbol → Click Analyze
- Get charts, signals, LSTM predictions, and AI insights

**RL Training:**
- Click Trading → Configure agent → Start Training
- Default: 300,000 steps (recommended for all algorithms)
- PPO/SAC/QRDQN: 15-20 min (300k steps)
- RecurrentPPO: 25-35 min (300k steps, LSTM needs more compute)
- Run Backtest → Compare strategies and metrics

**Live Trading:**
- Click Live Trade → Configure settings → Start Trading
- Monitor real-time portfolio, positions, and trades with virtual capital

**Model Management:**
- Click Models → View all trained LSTM and RL models

📖 **Detailed Guide**: See [QUICK_START.md](docs/QUICK_START.md) for step-by-step workflows

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│              Web Interface (Panel Dashboard)                          │
│  Light Theme • Wide Layouts • Responsive Design                       │
├──────────┬──────────┬──────────┬────────────┬──────────┬────────────┤
│Dashboard │ Analysis │ Training │ Live Trade │Watchlist │   Models   │
│• Markets │ • Charts │• RL Train│• Paper     │• Prices  │ • LSTM     │
│•Watchlist│ • Signals│• Backtest│• Real-time │• Symbols │ • RL       │
│• Actions │ • Predict│• Compare │• Risk Mgmt │• Simple  │ • Tabs     │
└──────────┴──────────┴──────────┴────────────┴──────────┴────────────┘
              │                          │
              ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│   Analysis Engine        │  │   RL Engine                  │
│   • Ollama AI            │  │   • Trading Environments     │
│   • LSTM Ensemble        │  │   • PPO/RecurrentPPO/SAC/    │
│   • Technical Indicators │  │     QRDQN Agents             │
│   • Chart Generation     │  │   • Backtest Engine          │
│                          │  │   • Baseline Strategies      │
└──────────────────────────┘  └──────────────────────────────┘
              │                          │
              └──────────┬───────────────┘
                         ▼
              ┌─────────────────────────┐
              │   Data Layer            │
              │   • Yahoo Finance       │
              │   • Intelligent Caching │
              │   • Model Storage       │
              └─────────────────────────┘
```

**Technology Stack:**
- **AI**: Ollama (gemma3:latest) with regex fallback
- **ML**: TensorFlow LSTM ensemble models (3 models per symbol)
- **RL**: Stable-Baselines3 (PPO, SAC) + sb3-contrib (RecurrentPPO, QRDQN), Gymnasium environments
- **Data**: Yahoo Finance API with intelligent caching
- **UI**: Panel + Plotly interactive visualizations, light theme
- **Design**: Wide horizontal layouts, minimal scrolling

---

## 📁 Project Structure

```
stock_agent_ollama/
├── src/                    # Core application
│   ├── agents/            # AI query processing
│   ├── rl/                # Reinforcement Learning
│   │   ├── env_factory.py
│   │   ├── model_utils.py
│   │   ├── sac_discrete_wrapper.py
│   │   ├── environments.py
│   │   ├── training.py
│   │   ├── backtesting.py
│   │   ├── live_trading.py
│   │   └── improvements.py
│   ├── tools/             # Data and analysis
│   │   ├── stock_fetcher.py
│   │   ├── technical_analysis.py
│   │   └── lstm/          # LSTM models
│   ├── ui/                # Web interface
│   │   ├── app.py
│   │   └── pages/
│   ├── config.py
│   └── main.py
│
├── data/                  # Data storage
│   ├── cache/            # Stock data
│   ├── models/           # Trained models
│   ├── logs/             # Application logs
│   └── live_sessions/    # Trading sessions
│
├── docs/                  # Documentation
│   ├── QUICK_START.md
│   ├── RL_DESIGN.md
│   └── LIVE_TRADE.md
│
└── requirements.txt       # Dependencies
```

---

## 📚 Documentation

### User Guides
- **[QUICK_START.md](docs/QUICK_START.md)** - Complete user guide with workflows and troubleshooting
- **[UX.md](docs/UX.md)** - Interface design, layouts, and component specifications

### Technical Documentation
- **[RL_DESIGN.md](docs/RL_DESIGN.md)** - RL architecture, algorithms, and design decisions
- **[LIVE_TRADE.md](docs/LIVE_TRADE.md)** - Live trading simulation with session persistence

---

## ⚙️ Configuration

**Environment Variables:**
```bash
OLLAMA_MODEL=gemma3:latest           # AI model for analysis
OLLAMA_BASE_URL=http://localhost:11434
PANEL_PORT=5006                      # Web interface port
RL_DEFAULT_INITIAL_BALANCE=100000.0  # Starting balance ($100k)
RL_TRANSACTION_COST_RATE=0.0005      # 0.05% transaction cost
RL_MAX_POSITION_PCT=80.0             # Max position size (80% of portfolio)
```

**Health Checks:**
```bash
# Test analysis engine
python -c "from src.agents.query_processor import QueryProcessor; print('✅ Ready')"

# Test RL engine
python -c "from src.rl import EnhancedRLTrainer, BacktestEngine; print('✅ Ready')"

# Check Ollama (optional)
curl http://localhost:11434/api/tags
```

---

## 🎨 Interface Features

### Light Theme Design
- Professional white/light gray color scheme
- High contrast for readability
- Color-coded values (green=positive, red=negative)
- Clean, modern aesthetics

### Wide Horizontal Layouts
- Minimal vertical scrolling
- Better use of widescreen monitors
- Dashboard: Markets + Quick Actions side-by-side
- Analysis: 70/30 split (Chart + Signals/Predictions)
- Optimized for desktop and laptop screens

### Live Data Updates
- Real-time market indices (5-second refresh)
- Live watchlist prices with color-coded changes
- Interactive charts with hover details
- Auto-loading of trained models

### Model Registry
- **LSTM Models**: Shows Final Loss, Val Loss, training date
- **RL Agents**: Shows algorithm, symbol, training date
- Performance data generated via backtesting
- Automatic model discovery and organization

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `source .venv/bin/activate` + `pip install -r requirements.txt` |
| Ollama unavailable | Platform works with fallback mode (regex-based) |
| Port 5006 in use | Set `PANEL_PORT=5007` |
| Memory issues | Need 8GB+ RAM for RL training |
| RL training slow | Default: 300k steps recommended for all algorithms. Can reduce to 100k for quick tests. |
| LSTM models empty | Run analysis once - auto-trains model (5-10 min) |
| RL shows "Run backtest →" | Normal - performance calculated during backtesting |

**More Help**: See [QUICK_START.md](docs/QUICK_START.md#troubleshooting)

---

## 📖 Key Features Explained

### LSTM Ensemble Predictions
- Trains 3 models per symbol for robustness
- 30-day price forecasts with confidence intervals
- Stored metadata includes Final Loss and Validation Loss
- Auto-training on first analysis request

### RL Trading Agents
- **PPO**: Proximal Policy Optimization (stable, reliable baseline)
- **RecurrentPPO**: LSTM-based PPO with trend indicators (best for downtrends)
  - Uses 13 features (10 base + 3 trend indicators)
  - SMA_Trend, EMA_Crossover, Price_Momentum indicators
  - Enhanced reward configuration optimized for trend-following
  - Automatic backwards compatibility with older models
- **SAC**: Soft Actor-Critic with continuous action space
  - DiscreteToBoxWrapper converts continuous outputs to 6 discrete actions
  - Maximum entropy framework encourages exploration
- **QRDQN**: Quantile Regression DQN (distributional RL for risk-aware decisions)
  - Off-policy learning with experience replay
  - Learns value distribution instead of expected value
- **Action Masking**: Prevents invalid trades (e.g., selling with no position)
- **6-Action Space**: HOLD (default), BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_PARTIAL, SELL_ALL
- **Adaptive Sizing**: Trade sizes adjust based on market volatility and portfolio state
- **Algorithm-Specific Rewards**: Optimized configs per algorithm
- **Environment Factory**: Single source of truth for configuration (env_factory.py)
- **Config Loading**: Live trading matches exact training environment
- Realistic environment with transaction costs and slippage

### Backtesting System
- Automatically loads all available trained models (PPO, RecurrentPPO, SAC, QRDQN)
- Detects and loads correct model type via model_utils
- Matches environment configuration to model training settings
- Compares all agents against Buy & Hold and Momentum baselines
- Comprehensive metrics: Returns, Sharpe, Sortino, Calmar ratios
- Action distribution visualization across strategies
- Clean performance comparison charts
- Key metrics bar charts for quick comparison

---

## 📖 Learning Objectives

**Stock Analysis & Prediction:**
- Technical indicators and their interpretation
- LSTM neural networks for time series forecasting
- Ensemble modeling for robust predictions
- AI-assisted financial analysis

**Reinforcement Learning Trading:**
- Policy optimization and actor-critic methods
- Trading environment design and reward engineering
- Risk-adjusted performance metrics
- Strategy comparison and evaluation
- Action space design for trading decisions

**Software Engineering:**
- Professional UX design patterns
- Real-time data handling and caching
- Model registry and management
- Wide layout optimization

---

## ⚠️ Important Disclaimer

**Educational Use Only**

This platform is designed for learning and research purposes:
- ✋ **NOT financial advice**
- ✋ **NOT for real trading decisions**
- ✋ Past performance does NOT guarantee future results
- ✋ RL agents trained on historical data may not work in live markets
- ✋ Always consult qualified financial professionals before making investment decisions

**Data Source**: Yahoo Finance (real-time, free, no API key required)

---

## 🔧 Key Technical Features

### Algorithm-Specific Configurations
- **RecurrentPPO**: LSTM memory with 3 trend indicators (SMA_Trend, EMA_Crossover, Price_Momentum)
  - `RecurrentPPORewardConfig`: Reduced penalties and momentum bonuses for trend-following
  - Hold Winner Bonus: Rewards holding profitable positions during uptrends
  - Momentum Trend Bonus: Additional reward for staying long during strong uptrends
  - 13-feature observation space (10 base + 3 trend indicators)
- **SAC**: Optimized for off-policy learning with continuous action discretization
  - `SACRewardConfig`: EXTREME reward shaping to overcome entropy-seeking (2025 v3)
    - **Base HOLD incentive**: +0.5 (was 0.05) - 10x stronger!
    - **Diversity PENALTY**: -1.0 for <30% diversity (prevents ALL collapse types)
    - **Consecutive penalty**: -1.0 base, scales to **-5.0 max** (was -1.5)
    - **Immediate** penalty on 1st repeat (no delay)
    - Transaction costs on ALL trades (prevents spam)
    - **Result**: BUY spam = -5.7, HOLD spam = -0.3, Mixed = +0.4 (forces diversity!)
  - Higher entropy coefficient (0.3) for action diversity
  - Adjusted training frequency for temporal stability
- **PPO**: Baseline on-policy algorithm
  - `PPORewardConfig`: Strong penalties and diversity bonuses to prevent action collapse
- **QRDQN**: Distributional RL for risk-aware decisions
  - `EnhancedRewardConfig`: Light penalties to let distributional learning work naturally

### Architecture Enhancements
- **Environment Factory Pattern**: Centralized configuration in `env_factory.py` ensures consistency across training, backtesting, and live trading
- **Model Utilities**: Automatic model type detection (PPO, RecurrentPPO, SAC, QRDQN) and environment config loading
- **SAC Wrapper**: DiscreteToBoxWrapper in `sac_discrete_wrapper.py` converts continuous actions to discrete for SAC
- **Single Source of Truth**: All default parameters defined once in `EnvConfig` dataclass
- **Conditional Features**: Trend indicators automatically enabled for RecurrentPPO, disabled for other algorithms

### Critical Bug Fixes
- **Short-Selling Prevention**: Fixed `AdaptiveActionSizer` bug that allowed unintentional short positions
- **Symbol Input**: Removed restrictions - now accepts any valid ticker symbol
- **Position Limits**: Corrected inconsistent defaults (now 80% across all systems)
- **Floating Point Precision**: Added tolerance to position size checks

### Live Trading Improvements
- **Multi-Session Support**: Run multiple trading strategies simultaneously
- **Session Persistence**: Auto-save and resume sessions across app restarts
- **Environment Matching**: Live trading automatically loads exact training configuration including trend indicators
- **UI Enhancements**: Wider model name column, improved session management

---

## 🎓 Perfect For

- **Students** learning AI/ML applications in finance
- **Researchers** exploring algorithmic trading strategies
- **Developers** studying RL implementations
- **Quantitative Analysts** experimenting with predictive models
- **Finance Professionals** understanding AI-driven analysis

Provides hands-on experience with modern algorithmic trading concepts in a safe, educational environment.

---

**Built for Financial AI & RL Education** 💙

*Comprehensive platform for learning stock analysis, LSTM predictions, and reinforcement learning trading strategies*