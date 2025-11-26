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
- **Train RL Agents** using PPO, RecurrentPPO, A2C, SAC, and QRDQN with action masking
- **6-Action Trading Space** (HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_PARTIAL, SELL_ALL)
- **RecurrentPPO** with LSTM memory for temporal pattern recognition
- **Trend Indicators** for RecurrentPPO (SMA_Trend, EMA_Crossover, Price_Momentum)
- **Adaptive Position Sizing** adjusts to market volatility
- **Training Metrics** (Win Rate, Action Distribution, Episode Rewards, Explained Variance)
- **Algorithm-Specific Rewards** optimized per algorithm
- **Comprehensive Backtesting** with automated model loading
- **Strategy Comparison** against Buy & Hold and Momentum
- **Performance Metrics** (Returns, Sharpe Ratio, Max Drawdown, Win Rate)
- **Visualization Charts** (Performance, Actions, Key metrics)

![RL Training Screenshot](docs/screenshots/training.png)
*Train and backtest RL agents with comprehensive performance metrics and strategy comparison*

### 🔴 Live Trading Simulation
- **Paper Trading** with real-time market data (Yahoo Finance)
- **Trained Agent Execution** using PPO, RecurrentPPO, A2C, SAC, or QRDQN models
- **Persistent Sessions** automatically save and resume
  - Portfolio state preserved
  - Trade history maintained
  - Configuration retained
- **Real-time Portfolio Tracking** with live P&L updates
- **Risk Management** (stop-loss, position limits, circuit breakers)
- **Live Monitoring** with status, positions, and event log
- **Educational Platform** for safe strategy testing

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
- Default: 300,000 steps (recommended)
- PPO/A2C/SAC/QRDQN: 15-20 min (300k steps)
- RecurrentPPO: 25-35 min (LSTM needs more compute)
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
│   • LSTM Ensemble        │  │   • PPO/RecurrentPPO/A2C/    │
│   • Technical Indicators │  │     SAC/QRDQN Agents         │
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
- **ML**: TensorFlow LSTM ensemble (3 models per symbol)
- **RL**: Stable-Baselines3 (PPO, A2C, SAC) + sb3-contrib (RecurrentPPO, QRDQN)
- **Data**: Yahoo Finance with intelligent caching
- **UI**: Panel + Plotly, light theme, wide layouts

---

## 📁 Project Structure

```
stock_agent_ollama/
├── src/
│   ├── agents/           # AI query processing
│   ├── rl/               # RL training, backtesting, live trading
│   ├── tools/            # Data fetching, technical analysis, LSTM
│   └── ui/               # Web interface (Panel)
├── data/
│   ├── cache/            # Stock data cache
│   ├── models/           # Trained models (LSTM, RL)
│   ├── logs/             # Application logs
│   └── live_sessions/    # Trading session persistence
├── docs/                 # User guides and technical docs
└── requirements.txt
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

**LSTM**: 3-model ensemble, 30-day forecasts, auto-training

**RL Agents**:
- **PPO**: Stable baseline with strong penalties
- **RecurrentPPO**: LSTM memory + trend indicators (13 features)
- **A2C**: Synchronous actor-critic with native discrete actions
- **SAC**: Maximum entropy off-policy with continuous-to-discrete wrapper
- **QRDQN**: Distributional RL for risk-awareness
- 6-action space with masking, adaptive sizing, algorithm-specific rewards

**Backtesting**: Auto-loads models, compares vs baselines, comprehensive metrics

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