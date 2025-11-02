# 🤖 Stock Analysis and Trading AI Platform

An intelligent financial platform combining **AI-powered analysis**, **LSTM neural networks**, and **reinforcement learning** for comprehensive stock analysis, predictions, and automated trading strategies.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![AI](https://img.shields.io/badge/AI-Ollama%20%2B%20LSTM%20%2B%20RL-orange)
![Educational](https://img.shields.io/badge/Purpose-Educational-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🎯 What It Does

### 📊 Stock Analysis & Prediction
- **Natural Language Queries** with Ollama AI
- **30-Day LSTM Predictions** using ensemble neural networks
- **Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
- **Interactive Visualizations** with real-time charts

![Stock Analysis Screenshot](docs/screenshots/analysis.png)
*Natural language stock analysis with AI-powered insights and LSTM predictions*

### 🎮 Reinforcement Learning Trading
- **Train RL Agents** using PPO and A2C algorithms
- **LSTM Hybrid Architecture** for temporal pattern extraction
- **Comprehensive Backtesting** with performance metrics
- **Strategy Comparison** against Buy & Hold and Momentum baselines

![RL Trading Screenshot](docs/screenshots/rl_trading.png)
*Train and backtest RL agents with comprehensive performance metrics*

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
**Stock Analysis Tab:**
- Ask questions: "Analyze Apple stock performance"
- Get LSTM predictions for next 30 days
- View technical indicators and trading signals

**RL Trading Tab:**
- Train agents on historical data (5-10 min for 50k steps)
- Backtest strategies with automatic model loading
- Compare RL agents vs traditional strategies

📖 **Detailed Guides**: See [QUICK_START.md](docs/QUICK_START.md) for step-by-step workflows

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Web Interface (Panel Dashboard)               │
├────────────────────────┬────────────────────────────────────┤
│   Stock Analysis       │        RL Trading                  │
│   • AI Queries         │        • Agent Training            │
│   • LSTM Predictions   │        • Backtesting               │
│   • Tech Analysis      │        • Strategy Comparison       │
└────────────────────────┴────────────────────────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│   Analysis Engine       │  │   RL Engine                  │
│   • Ollama AI           │  │   • Trading Environments     │
│   • LSTM Models         │  │   • PPO/A2C Agents           │
│   • Technical Indicators│  │   • Backtest Engine          │
└─────────────────────────┘  └──────────────────────────────┘
              │                           │
              └───────────┬───────────────┘
                          ▼
              ┌────────────────────────┐
              │   Data Layer           │
              │   • Yahoo Finance      │
              │   • File Caching       │
              │   • Model Storage      │
              └────────────────────────┘
```

**Technology Stack:**
- **AI**: Ollama (gemma3:latest) with regex fallback
- **ML**: TensorFlow LSTM ensemble models
- **RL**: Stable-Baselines3 (PPO, A2C), Gymnasium environments
- **Data**: Yahoo Finance API with intelligent caching
- **UI**: Panel + Plotly interactive visualizations

---

## 📁 Project Structure

```
stock_agent_ollama/
├── src/
│   ├── main.py                    # Application entry point
│   ├── agents/                    # AI query processing
│   ├── tools/                     # Stock data & analysis
│   │   └── lstm/                  # Neural network models
│   ├── rl/                        # Reinforcement Learning
│   │   ├── agents/                # PPO, A2C agents
│   │   ├── environments.py        # Trading environments
│   │   ├── training.py            # Training pipeline
│   │   ├── backtesting.py         # Backtest engine
│   │   ├── baselines.py           # Baseline strategies
│   │   ├── networks.py            # LSTM feature extractor
│   │   └── visualizer.py          # RL visualizations
│   └── ui/                        # Web interface
├── data/
│   ├── models/                    # Trained models (LSTM & RL)
│   ├── cache/                     # Stock data cache
│   └── conversations/             # Chat history
├── docs/
│   ├── QUICK_START.md             # Step-by-step user guide
│   ├── RL_DESIGN.md               # RL architecture & design
│   └── screenshots/               # README screenshots
└── requirements.txt
```

---

## 📚 Documentation

- **[QUICK_START.md](docs/QUICK_START.md)** - Detailed user guide with step-by-step workflows
- **[RL_DESIGN.md](docs/RL_DESIGN.md)** - Complete RL architecture, design decisions, and technical details

---

## ⚙️ Configuration

**Environment Variables:**
```bash
OLLAMA_MODEL=gemma3:latest
OLLAMA_BASE_URL=http://localhost:11434
PANEL_PORT=5006
RL_DEFAULT_INITIAL_BALANCE=10000.0
RL_TRANSACTION_COST_RATE=0.001
```

**Health Checks:**
```bash
# Test analysis engine
python -c "from src.agents.query_processor import QueryProcessor; print('✅ Ready')"

# Test RL engine
python -c "from src.rl import RLTrainer, BacktestEngine; print('✅ Ready')"

# Check Ollama (optional)
curl http://localhost:11434/api/tags
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `source .venv/bin/activate` |
| Ollama unavailable | Platform works with fallback mode |
| Port 5006 in use | Set `PANEL_PORT=5007` |
| Memory issues | Close apps, need 4GB+ for LSTM/RL training |
| RL training slow | Reduce timesteps or lookback window |

**More Help**: See [QUICK_START.md](docs/QUICK_START.md#troubleshooting)

---

## 📖 Learning Objectives

**Stock Analysis & Prediction:**
- Understand technical indicators and their interpretation
- Learn LSTM neural networks for time series forecasting
- Explore AI-assisted financial analysis

**Reinforcement Learning Trading:**
- Learn how RL agents optimize trading policies
- Understand policy optimization (PPO) vs actor-critic (A2C)
- Master risk-adjusted performance metrics
- Compare RL strategies with traditional approaches

---

**Built for Financial AI & RL Education** 💙

Perfect for students, researchers, and developers exploring AI and reinforcement learning applications in finance. Provides hands-on experience with modern algorithmic trading concepts in a safe, educational environment.
