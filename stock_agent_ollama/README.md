# 🤖 Stock Analysis and Trading AI Platform

An intelligent financial platform combining **AI-powered analysis**, **LSTM neural networks**, and **reinforcement learning** to provide comprehensive stock analysis, predictions, and automated trading strategies through an intuitive web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![AI](https://img.shields.io/badge/AI-Ollama%20%2B%20LSTM%20%2B%20RL-orange)
![Educational](https://img.shields.io/badge/Purpose-Educational-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🎯 What It Does

### 📊 Stock Analysis & Prediction
- **🤖 AI Query Processing**: Natural language questions powered by Ollama with gemma3:latest model
- **🧠 LSTM Predictions**: 30-day stock price forecasts using ensemble neural networks
- **📈 Technical Analysis**: 17+ indicators including RSI, MACD, Bollinger Bands, and trading signals
- **💬 Educational Explanations**: AI-generated interpretations of technical concepts
- **📉 Interactive Charts**: Real-time visualization with Plotly and Panel dashboard

### 🎮 Reinforcement Learning Trading
- **🚀 RL Agents**: Train intelligent trading agents using PPO and A2C algorithms
- **🧠 LSTM Features**: Optional LSTM feature extractor for temporal pattern recognition (hybrid architecture)
- **📊 Backtesting Engine**: Test strategies on historical data with comprehensive metrics
- **🎯 Action Analysis**: Visualize trading decisions (SELL/HOLD/BUY_SMALL/BUY_LARGE) with action distribution charts
- **⚖️ Strategy Comparison**: Auto-loads trained models and compares against baseline strategies (Buy & Hold, Momentum)
- **📈 Performance Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, win rate, and more
- **🎯 Custom Environments**: Realistic trading simulation with transaction costs and slippage

## 🚀 Quick Start

### 1. Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Ollama (Optional but Recommended)
```bash
# Install Ollama from https://ollama.ai
ollama pull gemma3:latest
```

### 3. Launch Platform
```bash
python src/main.py
# Open http://localhost:5006
```

## 💬 Usage Examples

### Stock Analysis Tab
**Natural Language Queries:**
```
"Analyze Apple stock performance"
"Predict Tesla price for next 30 days"
"Compare Microsoft vs Google"
"Explain what RSI means for beginners"
```

**Quick Actions:** Click buttons for AAPL, GOOGL, MSFT, TSLA, AMZN instant analysis

### RL Trading Tab
**Train RL Agents:**
1. Select stock symbol (AAPL, MSFT, GOOGL, etc.)
2. Choose algorithm (PPO or A2C)
3. Enable LSTM Features (optional, for temporal pattern extraction)
4. Set training period (30-730 days)
5. Configure training steps (10k-100k)
6. Click "Start Training" and watch the agent learn

**Backtest Strategies:**
1. Select stock symbol
2. Click "Run Backtest"
3. Automatically loads trained model if available
4. Compare RL agents against Buy & Hold and Momentum baselines
5. Analyze action distribution charts and performance metrics

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Panel)                     │
├────────────────────────┬────────────────────────────────────┤
│   Analysis Tab         │        RL Trading Tab              │
├────────────────────────┼────────────────────────────────────┤
│ • AI Query Processing  │ • Agent Training (PPO/A2C)         │
│ • LSTM Predictions     │ • Backtesting Engine               │
│ • Technical Analysis   │ • Strategy Comparison              │
│ • Interactive Charts   │ • Performance Analytics            │
└────────────────────────┴────────────────────────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│   Analysis Engine       │  │   RL Engine                  │
├─────────────────────────┤  ├──────────────────────────────┤
│ • Ollama AI Enhancer    │  │ • Trading Environments       │
│ • Hybrid Query Processor│  │ • RL Agents (Stable-Baselines3)│
│ • LSTM Models           │  │ • Reward Functions           │
│ • Technical Indicators  │  │ • Backtest Engine            │
└─────────────────────────┘  └──────────────────────────────┘
              │                           │
              └───────────┬───────────────┘
                          ▼
              ┌────────────────────────┐
              │   Data Layer           │
              ├────────────────────────┤
              │ • Yahoo Finance API    │
              │ • File-based Caching   │
              │ • Model Persistence    │
              └────────────────────────┘
```

### Technology Stack
- **AI**: Ollama (gemma3:latest) with regex fallback
- **ML**: TensorFlow LSTM ensemble models
- **RL**: Stable-Baselines3 (PPO, A2C), Gymnasium environments
- **Data**: Yahoo Finance API with intelligent caching
- **UI**: Panel framework with Plotly visualizations
- **Backend**: Python with asyncio and aiohttp

## 🎓 Educational Features

### Stock Analysis & Prediction
- **Technical Analysis**: Understand 17+ indicators like RSI, MACD, Bollinger Bands
- **Machine Learning**: Explore LSTM neural networks for time series prediction
- **Financial Concepts**: AI explanations adapted to your knowledge level
- **Market Analysis**: Learn to interpret trends, signals, and risk factors

### Reinforcement Learning Trading
- **RL Fundamentals**: Learn how agents learn optimal trading policies through trial and error
- **Policy Optimization**: Understand PPO (stable, sample-efficient) and A2C (faster, exploratory)
- **Environment Design**: See how trading environments model real market conditions
- **Performance Evaluation**: Master risk-adjusted metrics like Sharpe, Sortino, and Calmar ratios
- **Strategy Comparison**: Benchmark RL agents against traditional strategies

## 📁 Project Structure

```
stock_agent_ollama/
├── src/
│   ├── main.py                     # Application entry point
│   ├── agents/                     # AI & NLP processing
│   │   ├── query_processor.py      # Main query handler
│   │   ├── hybrid_query_processor.py  # Ollama integration
│   │   └── ollama_enhancer.py      # AI explanations
│   ├── tools/                      # Analysis & prediction
│   │   ├── lstm/                   # Neural network components
│   │   ├── stock_fetcher.py        # Data acquisition
│   │   ├── technical_analysis.py   # Indicators & signals
│   │   └── conversation_manager.py # Session management
│   ├── rl/                         # Reinforcement Learning
│   │   ├── agents/                 # RL agents (PPO, A2C)
│   │   │   ├── base_agent.py       # Base agent interface
│   │   │   ├── ppo_agent.py        # PPO agent implementation
│   │   │   └── a2c_agent.py        # A2C agent implementation
│   │   ├── environments.py         # Trading environments (Base + SingleStock)
│   │   ├── training.py             # Training pipeline (Trainer + Callbacks + Rewards)
│   │   ├── backtesting.py          # Backtesting (Engine + Metrics Calculator)
│   │   ├── baselines.py            # Baseline strategies (Buy&Hold, Momentum)
│   │   ├── networks.py             # Neural networks (LSTM feature extractor)
│   │   └── visualizer.py           # RL visualization tools
│   └── ui/                         # User interface
│       ├── components.py           # Main UI components
│       └── rl_components.py        # RL trading UI
├── data/                           # Runtime data
│   ├── models/
│   │   ├── lstm/                   # LSTM models
│   │   └── rl/                     # RL agent models
│   ├── cache/                      # Stock data cache
│   └── conversations/              # Chat sessions
└── requirements.txt                # Dependencies
```

## ⚙️ Configuration

### Environment Variables
```bash
# Ollama Settings
OLLAMA_MODEL=gemma3:latest
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_ENABLED=true

# Panel Settings
PANEL_PORT=5006
PANEL_HOST=localhost

# RL Settings
RL_DEFAULT_INITIAL_BALANCE=10000.0
RL_TRANSACTION_COST_RATE=0.001
RL_DEFAULT_TRAINING_TIMESTEPS=50000
```

### Health Checks
```bash
# Test core functionality
source .venv/bin/activate
python -c "from src.agents.query_processor import QueryProcessor; print('✅ Analysis engine ready')"

# Test RL components
python -c "from src.rl import RLTrainer, BacktestEngine; print('✅ RL engine ready')"

# Test Ollama integration
python -c "
import asyncio
from src.agents.ollama_enhancer import OllamaEnhancer

async def test():
    enhancer = OllamaEnhancer()
    healthy = await enhancer.health_check()
    print(f'Ollama: {\"✅ Connected\" if healthy else \"⚠️ Using fallback\"}')
    await enhancer.close()

asyncio.run(test())
"
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **Import errors** | Activate virtual environment: `source .venv/bin/activate` |
| **Ollama unavailable** | Platform works with regex fallback, install Ollama for AI features |
| **Port 5006 in use** | Set `PANEL_PORT=5007` environment variable |
| **Memory issues** | Close other applications, LSTM and RL training need 4GB+ RAM |
| **No predictions** | LSTM models train automatically on first prediction request |
| **RL training slow** | Reduce training timesteps or use smaller lookback window |
| **Backtest data missing** | Ensure internet connection for Yahoo Finance API |

### Common Commands
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test Panel interface
python -c "import panel as pn; print('Panel version:', pn.__version__)"

# View application logs
tail -f data/logs/app.log

# Test RL environment
python -c "
from src.rl import SingleStockTradingEnv
from datetime import datetime, timedelta

end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

env = SingleStockTradingEnv(
    symbol='AAPL',
    start_date=start_date,
    end_date=end_date
)
print(f'✅ Environment created with {env.max_steps} steps')
"
```

## 🎯 Key Features

### Hybrid Intelligence
- **AI-First**: Ollama processes complex natural language queries
- **Reliable Fallback**: Regex patterns ensure consistent functionality
- **Educational**: AI explanations make complex concepts accessible

### Advanced Analytics
- **Ensemble LSTM**: Multiple models for robust 30-day predictions
- **17+ Indicators**: Professional-grade technical analysis
- **Trading Signals**: BUY/SELL/HOLD recommendations with confidence
- **Risk Assessment**: Automated trend analysis and factor identification

### Reinforcement Learning
- **State-of-the-Art Algorithms**: PPO and A2C from Stable-Baselines3
- **Hybrid Architecture**: Optional LSTM feature extractor for temporal pattern recognition
- **Realistic Environments**: Transaction costs, slippage, and position limits
- **Multiple Reward Functions**: Simple returns, risk-adjusted, and customizable
- **Comprehensive Backtesting**: 15+ performance metrics including Sharpe, Sortino, Calmar ratios
- **Action Visualization**: See exactly what decisions agents make (SELL/HOLD/BUY_SMALL/BUY_LARGE)
- **Baseline Comparisons**: Auto-loads trained models and evaluates against traditional strategies

### User Experience
- **Tabbed Interface**: Separate workflows for analysis and RL trading
- **Real-time Progress**: Live training progress with episode statistics
- **Interactive Visualizations**: Plotly charts for all metrics and comparisons
- **Responsive Design**: Professional, compact UI that works on desktop and tablet

## 📊 RL Agent Training

### Quick Training Guide
1. **Select Parameters**:
   - Symbol: AAPL, MSFT, GOOGL, etc.
   - Algorithm: PPO (recommended) or A2C
   - LSTM Features: Enable for temporal pattern extraction (optional)
   - Training Period: 365 days (recommended)
   - Training Steps: 50,000 (5-10 minutes)

2. **Monitor Progress**:
   - Watch real-time episode count
   - Track training time
   - View progress bar

3. **Review Results**:
   - Training progress charts
   - Episode reward trends
   - Model save location

### Backtesting Workflow
1. **Run Backtest**: Tests strategies on last 6 months of data
2. **Auto-Load Model**: Automatically loads your most recent trained agent for the selected symbol
3. **Compare Metrics**: Side-by-side comparison table (RL Agent vs Buy & Hold vs Momentum)
4. **Action Analysis**: View action distribution table showing SELL/HOLD/BUY_SMALL/BUY_LARGE counts
5. **Analyze Charts**:
   - Portfolio value over time
   - Action distribution comparison (stacked bar chart)
   - Drawdown curves
   - Performance metrics comparison

### Performance Metrics Explained
- **Total Return**: Overall percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return vs. maximum drawdown
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## ⚖️ Educational License & Disclaimer

### Purpose
This platform is designed for **educational and research purposes**. It demonstrates:
- Modern AI techniques in financial analysis
- LSTM neural networks for time series forecasting
- Reinforcement learning for algorithmic trading
- Natural language processing for domain applications
- Professional software development practices

### Important Disclaimers
- 📚 **Educational Use Only** - Not for actual trading decisions
- ⚠️ **No Financial Advice** - AI and RL predictions are for learning purposes
- 🔬 **Research Tool** - Designed for academic and educational exploration
- 📊 **Past Performance** - Does not guarantee future results
- 🎮 **Simulated Trading** - RL agents trained on historical data only

### Risk Acknowledgment
- AI and RL predictions are experimental and educational
- Markets are unpredictable and can change rapidly
- Always consult qualified financial professionals
- Use paper trading to test strategies safely
- RL agents may not perform well in live market conditions

### Recommended Usage
- Study RL agent behavior in different market conditions
- Learn about reward engineering and policy optimization
- Understand the importance of backtesting and risk metrics
- Experiment with different training parameters safely
- Compare RL approaches with traditional strategies

---

**Built for Financial AI & RL Education** 💙

Perfect for students, researchers, and developers exploring AI and reinforcement learning applications in finance. Combines cutting-edge machine learning with educational best practices to provide hands-on experience with modern algorithmic trading concepts.
