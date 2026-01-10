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
  - **Top Movers Suggestions**: Automatically finds 8 high-momentum stocks (30-day max returns) from curated universe
  - Manual refresh button to force fresh market data on demand
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
- **News Sentiment Integration** for sentiment-aware price forecasting
- **Direct Horizon Models** for improved long-term accuracy
- **AI-Powered Analysis** with natural language insights
- **Trading Signals** (BUY/SELL/HOLD) with confidence scores

![Stock Analysis Screenshot](docs/screenshots/analysis.png)
*Stock analysis with interactive charts, technical indicators, and AI-powered insights*

### 🤖 Reinforcement Learning Trading
- **Train RL Agents** using PPO, RecurrentPPO, and Ensemble
- **Action Masking** prevents invalid trades automatically during execution
- **6-Action Trading Space** (HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_PARTIAL, SELL_ALL)
- **RecurrentPPO** with LSTM memory for temporal pattern recognition
- **Advanced Risk Management** (stop-loss, trailing stops, circuit breakers)
- **Market Regime Detection** (BULL, BEAR, SIDEWAYS, VOLATILE)
- **Multi-Timeframe Features** (weekly/monthly trend analysis)
- **Kelly Position Sizing** (optimal position sizing based on edge)
- **Ensemble Agents** (combine multiple algorithms)
- **Adaptive Position Sizing** adjusts to market volatility
- **Training Metrics** (Win Rate, Action Distribution, Episode Rewards, Explained Variance)
- **Algorithm-Specific Rewards** optimized per algorithm
- **Comprehensive Backtesting** with automated model loading
- **Strategy Comparison** against Buy & Hold and Momentum
- **Performance Metrics** (Returns, Sharpe Ratio, Max Drawdown, Win Rate)
- **Visualization Charts** (Performance, Price with Trade Signals, Actions, Key metrics)

![RL Training Screenshot](docs/screenshots/training.png)
*Train and backtest RL agents with comprehensive performance metrics and strategy comparison*

### 🔴 Live Trading Simulation (Paper Trading)
- **Paper Trading** with real-time market data (Yahoo Finance)
- **Trained Agent Execution** using PPO, RecurrentPPO, or Ensemble models
- **Realistic Transaction Costs** (zero-commission era: $0 commissions + 0.05% slippage)
  - Reflects modern brokers (Fidelity, Schwab, Robinhood)
  - Buy costs shown as negative P&L in trade record
  - Sell costs subtracted from realized P&L
  - Costs match backtesting environment exactly
  - Full transparency: costs displayed in trades table and session stats
- **Auto Stock Selection** dynamically rotates to best performing stocks
  - Dynamic scoring based on backtest performance (Sharpe ratio + returns)
  - Prioritizes agent backtest metrics, falls back to 5-day price performance
  - Intelligent idle detection: automatically rotates away from stocks where agent refuses to trade
  - **Active Signal Scan & Idle Detection**: Prevents capital stagnation by actively scanning for BUY signals when the current agent is idle
  - **Recency Penalty**: Prevents rapid "ping-pong" rotation by penalizing stocks rotated away from within the last 30 minutes
  - Rotation cooldown (10 cycles/10 minutes) to allow fair trading opportunities
  - Performance threshold (2% improvement) to prevent unnecessary rotation
  - Automatically closes positions before rotation for optimal opportunity capture
  - Single AUTO session policy prevents duplicate sessions and capital fragmentation
  - Maximizes capital efficiency across portfolio
- **Persistent Sessions** automatically save and resume
  - Portfolio state preserved
  - Trade history maintained
  - Configuration retained
  - Sessions sorted by creation time (newest first)
- **Real-time Portfolio Tracking** with live P&L updates
- **Risk Management** (stop-loss, position limits, circuit breakers)
- **Churn Protection** prevents rapid ping-pong trading (15-min cooldown)
- **Live Monitoring** with status, positions, and event log showing symbol-prefixed events
- **Optimized Dashboard** with granular updates for high-performance multi-session monitoring
- **Educational Platform** for safe strategy testing

![Live Trading Screenshot](docs/screenshots/live_trade.png)
*Real-time paper trading with persistent sessions, portfolio tracking, and risk management*

### 🗂️ Model Registry
- **LSTM Models** with performance metrics (Final Loss, Val Loss)
- **RL Agents** with training dates and algorithm types
- **Batch Backtesting** with checkbox selection for multiple models
- **Model Management** with automatic discovery and organization
- **Chronological Ordering** with newest models displayed first for both LSTM and RL agents

![Model Registry Screenshot](docs/screenshots/models.png)
*Model registry with LSTM and RL agent listings, batch backtesting with checkbox selection*

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
- Default: 300,000 steps (recommended)
- PPO: 15-20 min (300k steps)
- RecurrentPPO: 25-35 min (LSTM needs more compute)
- Ensemble: 40-55 min (trains both PPO + RecurrentPPO)
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
│   • LSTM Ensemble        │  │   • PPO/RecurrentPPO/        │
│   • Technical Indicators │  │     Ensemble Agents          │
│   • Chart Generation     │  │   • Backtest Engine          │
│                          │  │   • Baseline Strategies      │
└──────────────────────────┘  └──────────────────────────────┘
              │                          │
              └──────────┬───────────────┘
                         ▼
              ┌─────────────────────────┐
              │   Data Layer            │
              │   • Yahoo Finance       │
              │   • Multi-tier Caching  │
              │     - Real-time: 1 min  │
              │     - Bulk/Top: 5 min   │
              │     - Info: 1 hour      │
              │     - Historical: 1 day │
              │   • Model Storage       │
              │   • Session Persistence │
              └─────────────────────────┘
```

**Technology Stack:**
- **AI**: Ollama (gemma3:latest) with regex fallback
- **ML**: TensorFlow LSTM ensemble (3 models per symbol)
- **RL**: Stable-Baselines3 (PPO) + sb3-contrib (RecurrentPPO) + Custom Ensemble
- **Data**: Yahoo Finance with intelligent multi-tier caching system
  - Real-time quotes: 1 minute cache for live price data
  - Intraday data: 15 minutes for 1m/5m intervals
  - Bulk data/Top Movers: 5 minutes for optimal performance
  - Company fundamentals: 1 hour for stable information
  - Historical data: 1 day for long-term charts
- **UI**: Panel + Plotly, light theme, wide layouts

---

## 📁 Project Structure

```
stock_agent_ollama/
├── src/
│   ├── agents/           # AI query processing
│   ├── rl/               # RL training, backtesting, live trading
│   ├── tools/            # Data fetching, technical analysis, LSTM
│   ├── ui/               # Web interface (Panel)
│   └── config.py         # Centralized configuration
├── data/
│   ├── cache/            # Stock data cache
│   ├── models/           # Trained models (LSTM, RL)
│   │   ├── archive/      # Archived models
│   │   ├── lstm/         # LSTM models
│   │   └── rl/           # RL agent models
│   ├── logs/             # Application logs
│   └── live_sessions/    # Trading sessions
├── tests/                # Test suite (135 tests, 19% coverage)
├── docs/                 # Documentation
├── retrain_rl.py         # Training automation
├── validate_backtest.py  # Backtest validation
├── eval_training.py      # Model evaluation and analysis
└── requirements.txt      # Dependencies
```

---

## 📚 Documentation

### User Guides
- **[QUICK_START.md](docs/QUICK_START.md)** - Complete user guide with workflows and troubleshooting
- **[UX.md](docs/UX.md)** - Interface design, layouts, and component specifications

### Technical Documentation
- **[RL_DESIGN.md](docs/RL_DESIGN.md)** - RL architecture, algorithms, and design decisions
- **[LIVE_TRADE.md](docs/LIVE_TRADE.md)** - Live trading simulation with session persistence
- **[TESTING.md](docs/TESTING.md)** - Comprehensive testing guide with 135 tests (19% coverage, critical modules 94-100%)

---

## ⚙️ Configuration

**Environment Variables:**
```bash
OLLAMA_MODEL=gemma3:latest           # AI model for analysis
OLLAMA_BASE_URL=http://localhost:11434
PANEL_PORT=5006                      # Web interface port

# Data Caching (intelligent multi-tier system)
CACHE_TTL_SECONDS=3600               # Default cache TTL (1 hour)
REALTIME_DATA_TTL=60                 # Real-time quotes (1 minute)
STOCK_DATA_TTL=900                   # Intraday data (15 minutes)
BULK_DATA_TTL=300                    # Bulk data/Top Movers (5 minutes)
STOCK_INFO_TTL=3600                  # Company fundamentals (1 hour)
HISTORICAL_DATA_TTL=86400            # Historical OHLCV data (1 day)

# RL Trading Configuration
RL_DEFAULT_INITIAL_BALANCE=100000.0  # Starting balance ($100k)
RL_TRANSACTION_COST_RATE=0.0         # $0 commissions (zero-commission era)
RL_SLIPPAGE_RATE=0.0005              # 0.05% slippage for liquid stocks
RL_DEFAULT_TRAINING_TIMESTEPS=300000 # Default training steps (recommended)
RL_MAX_POSITION_PCT=80.0             # Max position size (80% of portfolio)
RL_STOP_LOSS_PCT=0.05                # 5% stop-loss (default)
RL_TRAILING_STOP_PCT=0.03            # 3% trailing stop (default)
RL_MAX_DRAWDOWN_PCT=0.15             # 15% max drawdown (default)
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

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `source .venv/bin/activate && pip install -r requirements.txt` |
| Port in use | Set `PANEL_PORT=5007` |
| Memory issues | Need 8GB+ RAM for RL training |
| Training slow | 300k steps recommended, reduce to 100k for quick tests |

See [QUICK_START.md](docs/QUICK_START.md#troubleshooting) for more help

---

## 📖 Key Features

**LSTM Predictions:**
- 3-model ensemble for robustness
- 30-day price forecasts
- Auto-training on first analysis

**RL Agents:**
- **PPO**: Exceptional performance with balanced risk management (2.81 Sharpe ratio)
- **RecurrentPPO**: LSTM memory + trend indicators for temporal pattern recognition (1.84 Sharpe ratio)
- **Ensemble**: Weighted voting combining PPO (30%) + RecurrentPPO (70%) (1.89 Sharpe ratio)
- Action masking ensures only valid actions are executed
- 6-action space with adaptive sizing and algorithm-specific rewards

**Backtesting:**
- Auto-loads models
- Compares vs baselines (Buy & Hold, Momentum)
- Comprehensive metrics (Sharpe, Max Drawdown, Win Rate)

**Comprehensive Test Suite:**
The project includes 138 automated tests covering core functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_action_masking.py -v
```

**Test Coverage (19% overall, critical modules 94-100%):**
- **Technical Analysis (99%)** - All indicators (RSI, MACD, Bollinger Bands, Stochastic, ATR, trend analysis, trading signals)
- **Reward Functions (94%)** - All reward calculation strategies (Simple, RiskAdjusted, Customizable)
- **Baseline Strategies (100%)** - Buy & Hold, Momentum strategies with full edge case coverage
- **Configuration (96%)** - Settings, environment variables, directory management
- **Action Masking (100%)** - Invalid trade prevention validation
- **Live Trading Models (100%)** - Portfolio, Position, Trade, Order data structures
- **RL Components (90%)** - Ensemble voting, environment factory
- **LSTM Components** - Prediction service, data pipeline, and sentiment integration validation

**Test Files:**
- `tests/test_lstm_components.py` - LSTM prediction and sentiment integration (3 tests)
- `tests/test_reward_functions.py` - RL reward calculations (29 tests)
- `tests/test_baseline_strategies.py` - Trading baselines (35 tests)
- `tests/test_config.py` - Configuration validation (14 tests)
- `tests/test_live_trading_models.py` - Data models (14 tests)
- `tests/test_action_masking.py` - Action masking (10 tests)
- `tests/test_rl_components.py` - RL components (5 tests)
- `tests/test_technical_analysis.py` - All technical indicators (28 tests)



## 🛠️ Developer Tools

The project includes a suite of CLI tools for the full RL lifecycle:

**`retrain_rl.py`** - Training automation and comprehensive backtesting
- Train RL agents (PPO, RecurrentPPO, Ensemble)
- Run comprehensive backtests comparing all algorithms
- Compare multiple model versions for tracking improvements

**`retrain_and_predict.py`** - LSTM maintenance and validation
- Retrains LSTM ensemble for a specific symbol
- Generates fresh prediction for immediate verification
- Validates prediction against current price (gap check)
- Supports specific prediction horizons (e.g., h=30)
- Supports batch processing for watchlist symbols

**`validate_backtest.py`** - Backtest validation and integrity checks
- Validates backtest results across all trained models
- Runs 10 comprehensive checks per model:
  - Return calculation, action distribution, win rate
  - Portfolio consistency, metrics reasonableness
  - Individual trade P&L, transaction costs
  - Return reconciliation from trades
  - Market data integrity, reproducibility
- Supports watchlist-wide validation or single symbol validation
- Shows profitability rates and pass/fail summaries

**`eval_training.py`** - Model performance analysis and insights
- Scans and analyzes all trained RL models
- Shows top 5 and bottom 5 performers with detailed metrics
- Sort by return, Sharpe ratio, win rate, max drawdown, or age
- Provides training insights (profitability rate, average returns, Sharpe ratios, algorithm comparisons)
- Detects training pathologies (action collapse, overtrading, poor risk-adjusted returns)
- Supports model pruning to archive underperforming models
- Filter by symbol, algorithm, or performance thresholds

👉 **For detailed usage and command examples, see [QUICK_START.md](docs/QUICK_START.md#developer-tools).**

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

## 🎓 Perfect For

Students, researchers, developers, quants, and finance professionals learning AI/ML trading in a safe educational environment.

---

**Built for Financial AI & RL Education** 💙

*Comprehensive platform for learning stock analysis, LSTM predictions, and reinforcement learning trading strategies*