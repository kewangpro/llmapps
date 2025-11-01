# Stock Analysis and Trading AI - Quick Start Guide

## 🚀 Getting Started

### Installation

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the application
python src/main.py
```

The application will open at: **http://localhost:5006**

---

## 📊 Using Stock Analysis

### Quick Analysis (2 clicks)
1. Click the **📊 Analysis** tab
2. Click a quick stock button (AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META)
3. View results instantly!

### Custom Queries
Type natural language queries in the input box:
- `"Analyze AAPL"`
- `"Predict GOOGL for 30 days"`
- `"Compare MSFT vs AAPL"`
- `"What is TSLA current price?"`
- `"Explain RSI for beginners"`

### Results You'll See
- **Stock Info Card**: Company name, current price, daily change, market cap
- **Interactive Chart**: Price history with technical indicators (hover for details)
- **Trading Signal**: BUY/SELL/HOLD recommendation with confidence level
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and 15+ indicators
- **AI Analysis**: Natural language explanation of trends and signals
- **30-Day Prediction**: LSTM neural network forecast (if model is trained)

---

## 🤖 Using RL Trading

### Training an Agent (5-10 minutes)

1. **Navigate**: Click **🤖 RL Trading** tab
2. **Configure**:
   - **Symbol**: Select stock from dropdown (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, ORCL)
   - **Algorithm**: Choose PPO (recommended, stable) or A2C (faster, experimental)
   - **Training Days**: 365 days (default, recommended)
   - **Training Steps**: 50,000 (default, 5-10 minutes training time)
3. **Train**: Click "🚀 Start Training"
4. **Monitor**: Watch the progress bar
5. **Results**:
   - Training summary card with episodes, time, and model path
   - Training progress chart showing reward improvement
   - Model automatically saved to `data/models/rl/`

### Running Backtests

1. **Select Stock**: Choose symbol from dropdown (top of RL Trading tab)
2. **Click**: "📊 Run Backtest"
3. **Wait**: ~30 seconds
4. **Results**:
   - Comparison table (Buy & Hold vs Momentum strategies)
   - Performance metrics: Return, Sharpe ratio, Max drawdown, Win rate
   - Performance comparison chart (portfolio value over time)
   - Metrics bar chart (side-by-side comparison)

### Understanding Results

**Training Results:**
- **Agent**: Algorithm type (PPO or A2C)
- **Stock**: Symbol trained on
- **Episodes**: Number of complete trading episodes
- **Time**: Total training duration in seconds
- **Model Path**: Where the trained model is saved
- **Chart**: Episode rewards over time (shows learning progress)

**Backtest Metrics:**
- **Return**: Total profit/loss percentage over test period
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)
- **Win Rate**: Percentage of profitable trades

---

## 💡 Tips & Tricks

### Stock Analysis
- ✅ Use quick buttons for fastest analysis
- ✅ Charts are interactive - hover over points for exact values
- ✅ Green signals = bullish, Red = bearish, Orange = neutral
- ✅ LSTM predictions require trained models (train with "Predict [SYMBOL]")
- ✅ AI explanations adapt to query context

### RL Trading
- ✅ **PPO** is recommended for stable, reliable strategies
- ✅ **A2C** is faster for quick experiments
- ✅ **365 days** of training data balances learning vs. overfitting
- ✅ **50,000 steps** is a good default (balance of time vs. performance)
- ✅ Training runs in background - UI stays responsive
- ✅ Models are saved automatically - reusable across sessions
- ✅ Backtests compare strategies on last 6 months of data

### Performance
- ⚡ Stock data caches for faster repeated queries
- ⚡ Training is CPU-intensive - reduce steps for faster testing
- ⚡ Backtests are fast (~30 seconds per symbol)
- ⚡ LSTM predictions load existing models (fast) or train new ones (5-10 min)

---

## 🎯 Example Workflows

### Workflow 1: Quick Stock Check (5 seconds)
```
1. Click "📊 Analysis" tab
2. Click "AAPL" quick button
3. Review current price and signals
4. Scroll down to see AI analysis
```

### Workflow 2: Deep Analysis with Prediction (10 seconds)
```
1. Type: "Predict TSLA for 30 days"
2. Review current technical indicators
3. Check 30-day LSTM forecast chart
4. Read AI analysis of trends
```

### Workflow 3: Compare Multiple Stocks (15 seconds)
```
1. Type: "Compare MSFT vs GOOGL vs NVDA"
2. Review side-by-side metrics
3. Compare performance charts
4. Check which has better momentum
```

### Workflow 4: Train Your First RL Agent (10 minutes)
```
1. Click "🤖 RL Trading" tab
2. Select stock: NVDA (from dropdown)
3. Choose algorithm: PPO
4. Keep defaults: 365 days, 50,000 steps
5. Click "🚀 Start Training"
6. Wait 5-10 minutes (watch progress)
7. Review training chart and results
```

### Workflow 5: Backtest Strategies (30 seconds)
```
1. RL Trading tab
2. Select stock: AAPL
3. Click "📊 Run Backtest"
4. Compare Buy & Hold vs Momentum
5. Review metrics: returns, Sharpe, drawdown
6. Analyze performance charts
```

---

## 🔍 Understanding the Interface

### Main Layout
```
┌─────────────────────────────────────────────────────┐
│  Stock Analysis and Trading AI  (browser header)    │
├─────────────────────────────────────────────────────┤
│  [📊 Analysis] [🤖 RL Trading]  ← Tabs              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Tab content appears here                           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Analysis Tab
```
┌─────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────┐ │
│ │ [Search input___________________] [Analyze] ⟳   │ │
│ │ [AAPL] [GOOGL] [MSFT] [TSLA] [AMZN] [NVDA] ... │ │
│ └─────────────────────────────────────────────────┘ │ ← Input controls
├─────────────────────────────────────────────────────┤
│                                                      │
│  📈 Stock Info Card                                 │
│  📊 Interactive Price Chart                         │
│  🎯 Trading Signal                                  │
│  📈 30-Day Prediction (if available)                │
│  🤖 AI Analysis                                     │
│                                                      │
├─────────────────────────────────────────────────────┤
│ ⚠️ Educational Disclaimer                           │
└─────────────────────────────────────────────────────┘
```

### RL Trading Tab
```
┌─────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────┐ │
│ │  Symbol        Algorithm                        │ │
│ │  [AAPL ▾]      [PPO] [A2C]                      │ │
│ │                                                  │ │
│ │  Training Period: ──●────── 365 days            │ │
│ │  Training Steps:  ──●────── 50000               │ │
│ │                                                  │ │
│ │  [🚀 Start Training]  [📊 Run Backtest]         │ │
│ │  Progress: ████████░░ 80%                       │ │
│ └─────────────────────────────────────────────────┘ │ ← Controls
├─────────────────────────────────────────────────────┤
│                                                      │
│  Training/Backtest Results:                         │
│  - Summary cards                                    │
│  - Progress charts                                  │
│  - Metrics tables                                   │
│  - Comparison visualizations                        │
│                                                      │
├─────────────────────────────────────────────────────┤
│ ⚠️ Educational Disclaimer                           │
└─────────────────────────────────────────────────────┘
```

---

## 📚 Key Concepts

### Stock Analysis
- **Technical Indicators**: Mathematical calculations on price/volume data
  - **RSI** (0-100): Oversold (<30) or Overbought (>70)
  - **MACD**: Trend strength and direction
  - **Bollinger Bands**: Volatility and price extremes
  - **Moving Averages**: Trend identification

- **Trading Signals**: BUY/SELL/HOLD recommendations based on multiple indicators
- **LSTM Predictions**: Neural network forecasts trained on 2 years of data

### RL Trading
- **RL Agent**: AI that learns to trade through trial and error
  - **PPO** (Proximal Policy Optimization): Stable, sample-efficient
  - **A2C** (Advantage Actor-Critic): Faster, good for experiments

- **Training**: Agent practices thousands of trading episodes on historical data
- **Environment**: Simulates realistic trading with transaction costs and slippage
- **Reward**: Agent learns to maximize risk-adjusted returns

- **Baseline Strategies**:
  - **Buy & Hold**: Buy on day 1, hold until end
  - **Momentum**: Buy on uptrends, sell on downtrends

### Performance Metrics
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Return per unit of risk (>1 good, >2 excellent)
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: Return divided by max drawdown
- **Max Drawdown**: Worst peak-to-trough decline (risk measure)
- **Win Rate**: Percentage of profitable trades

---

## ⚠️ Important Notes

### Educational Use Only
- ✋ This platform is for **learning and research**
- ✋ **NOT financial advice**
- ✋ **NOT for real trading decisions**
- ✋ Past performance does NOT guarantee future results
- ✋ RL agents trained on historical data may not work in live markets
- ✋ Always consult qualified financial professionals

### Data Sources
- **Yahoo Finance**: Real-time and historical stock data
- **Free**: No API key required
- **Coverage**: Major US stocks and indices
- **Updates**: Automatic data fetching with intelligent caching

### System Requirements
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB+ recommended (for RL training)
- **Storage**: 1GB+ (for models, data cache, and logs)
- **Internet**: Required for fetching stock data
- **CPU**: Multi-core recommended for faster RL training

---

## 🐛 Troubleshooting

### "Training failed" or "Insufficient data"
**Cause**: Not enough historical data for the selected stock
**Fix**:
- Choose established stocks (AAPL, MSFT, GOOGL, AMZN, TSLA)
- Reduce training period to 180 days
- Check internet connection

### "No trained models found for [SYMBOL]"
**Cause**: LSTM model hasn't been trained for this symbol
**Fix**:
- Use explicit prediction query: "Predict [SYMBOL]" to auto-train
- Training takes 5-10 minutes on first request
- Models save automatically for reuse

### Charts Not Displaying
**Cause**: Missing Plotly dependency or browser compatibility
**Fix**:
- Run: `pip install plotly`
- Try different browser (Chrome recommended)
- Clear browser cache

### Slow RL Training
**Cause**: High number of timesteps or CPU limitations
**Fix**:
- Reduce to 30,000 steps for testing
- Reduce training period to 180 days
- Close other applications
- Use PPO (generally faster than A2C)

### "Module not found" Error
**Cause**: Missing dependencies
**Fix**:
- Activate virtual environment: `source .venv/bin/activate`
- Run: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.9+)

### Memory Errors During Training
**Cause**: Insufficient RAM
**Fix**:
- Close other applications
- Reduce training steps to 30,000
- Reduce training period to 180 days
- Reduce lookback window in config

---

## 🎓 Learn More

### Documentation
- **README.md**: Complete project overview, features, architecture
- **RL_DESIGN.md**: Detailed RL system design and components
- **UI_IMPROVEMENTS.md**: UI architecture and design decisions

### Key Files
- **Models**: `data/models/lstm/` (LSTM models) and `data/models/rl/` (RL agents)
- **Cache**: `data/cache/stock_data/` (cached stock data)
- **Logs**: `data/logs/app.log` (application logs)
- **Config**: `src/config.py` (configuration settings)

---

## 🆘 Need Help?

### Debugging Steps
1. Check application logs: `tail -f data/logs/app.log`
2. Verify dependencies: `pip list | grep -E "panel|tensorflow|stable-baselines3"`
3. Test basic functionality: Run health checks from README
4. Review error messages in browser console (F12)

### Resources
- **RL Questions**: See RL_DESIGN.md for architecture details
- **UI Questions**: See UI_IMPROVEMENTS.md for interface design
- **Code Issues**: Check GitHub issues or create a new one
- **General Help**: Review this guide and README.md

---

## 🎉 You're Ready!

### Recommended First Steps

**Beginner Path:**
1. 📊 Click "AAPL" button to see instant analysis
2. 🔍 Try typing "Predict TSLA" to see forecasts
3. 📊 Switch to RL Trading and click "Run Backtest"
4. 📖 Review the results and learn the metrics

**Advanced Path:**
1. 🤖 Train your first RL agent (PPO, NVDA, 365 days)
2. 📊 Run backtest to compare strategies
3. 🔬 Experiment with different stocks and algorithms
4. 📈 Compare multiple training configurations

**Tips for Learning:**
- Start with familiar stocks (AAPL, MSFT)
- Read the disclaimers at bottom of each tab
- Experiment with different query styles
- Compare RL strategies to understand performance metrics
- Use smaller timesteps (30k) for faster experiments

---

**Enjoy exploring AI-powered stock analysis and reinforcement learning trading!** 📈🤖

*Remember: This is for education only. Never use AI predictions for real trading without consulting financial professionals.*
