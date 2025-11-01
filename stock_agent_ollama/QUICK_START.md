# Stock Analysis AI - Quick Start Guide

## 🚀 Getting Started

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the application
python src/main.py
```

The application will open at: **http://localhost:5006**

---

## 📊 Using Stock Analysis

### Quick Analysis (2 clicks)
1. Click a quick stock button (AAPL, GOOGL, etc.)
2. View results instantly!

### Custom Queries
Type natural language queries:
- `"Analyze AAPL"`
- `"Predict GOOGL for 30 days"`
- `"Compare MSFT vs AAPL"`
- `"What is TSLA current price?"`

### Results You'll See
- **Header Card**: Stock name, current price, daily change
- **Chart**: Interactive price chart with technical indicators
- **Signal Card**: BUY/SELL/HOLD recommendation with confidence
- **Prediction Card**: 30-day forecast (if available)
- **AI Analysis**: Detailed analysis (click to expand)

---

## 🤖 Using RL Trading

### Training an Agent (1 minute setup)

1. **Navigate**: Click **🤖 RL Trading** tab
2. **Configure**:
   - Stock: AAPL (or choose from dropdown)
   - Algorithm: PPO (recommended) or A2C
   - Training Days: 365 (default)
   - Training Steps: 50,000 (default)
3. **Train**: Click "🚀 Start Training"
4. **Wait**: 5-10 minutes (watch progress bar)
5. **Results**: View training progress chart and episode stats

### Running Backtests

1. **Select Stock**: Choose symbol from dropdown
2. **Click**: "📊 Run Backtest"
3. **Wait**: ~30 seconds
4. **Results**:
   - Metrics table (Return, Sharpe, Max Drawdown, Win Rate)
   - Performance comparison chart
   - Metrics bar charts

### Understanding Results

**Training Results:**
- **Episodes**: Number of trading episodes completed
- **Time**: Total training duration
- **Chart**: Reward progression over time

**Backtest Metrics:**
- **Return**: Total profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

---

## 💡 Tips & Tricks

### Stock Analysis
- ✅ Use quick buttons for instant analysis
- ✅ Collapse AI analysis if you just want numbers
- ✅ Charts are interactive - hover for details
- ✅ Green = bullish, Red = bearish, Orange = neutral

### RL Trading
- ✅ **PPO** for stable, reliable strategies
- ✅ **A2C** for faster experimentation
- ✅ **365 days** of training data is a good default
- ✅ **50,000 steps** balances time vs. performance
- ✅ Train during breaks - it takes 5-10 minutes
- ✅ Backtest compares your strategy to baselines

### Performance
- ⚡ Results cache for 15 minutes (faster repeated queries)
- ⚡ Training runs in background (UI stays responsive)
- ⚡ Backtests are fast (~30 seconds)

---

## 🎯 Example Workflows

### Workflow 1: Quick Stock Check
```
1. Click "AAPL" quick button
2. Review price and signals
3. Expand AI analysis if needed
Time: 5 seconds
```

### Workflow 2: Deep Analysis
```
1. Type: "Analyze TSLA for 6 months"
2. Review technical indicators
3. Check prediction
4. Read AI analysis
Time: 10 seconds + reading time
```

### Workflow 3: Train Trading Agent
```
1. Switch to RL Trading tab
2. Select stock: NVDA
3. Choose algorithm: PPO
4. Set training days: 365
5. Click "Start Training"
6. Get coffee ☕
7. Review results in 5-10 min
```

### Workflow 4: Compare Strategies
```
1. RL Trading tab
2. Select stock: AAPL
3. Click "Run Backtest"
4. Compare Buy & Hold vs Momentum
5. Review metrics table
Time: 30 seconds
```

---

## 🔍 Understanding the Interface

### Header
```
┌──────────────────────────────────────────┐
│ 📈 Stock Analysis AI         v1.0.0     │
│ LSTM • Technical Analysis • RL Trading   │
└──────────────────────────────────────────┘
```

### Analysis Tab
```
┌──────────────────────────────────────────┐
│ 📊 Analysis | 🤖 RL Trading             │ ← Tabs
├──────────────────────────────────────────┤
│ [Query Input____________] [Analyze]      │
│ [AAPL] [GOOGL] [MSFT] [TSLA] ...        │ ← Quick buttons
├──────────────────────────────────────────┤
│ Results appear here                      │
└──────────────────────────────────────────┘
```

### RL Trading Tab
```
┌──────────────────────────────────────────┐
│ 📊 Analysis | 🤖 RL Trading             │
├──────────────────────────────────────────┤
│ ℹ️ About RL Trading                      │
│ [Info card with explanation]             │
├──────────────────────────────────────────┤
│ ⚙️ Settings ▼                            │
│  Stock: [AAPL ▾]  Algorithm: [PPO] [A2C] │
│  Training Days: ──●────── 365            │
│  Training Steps: ──●────── 50000         │
│  [🚀 Start Training] [📊 Run Backtest]   │
│  Progress: ████░░░░░ 45%                 │
├──────────────────────────────────────────┤
│ Results appear here                      │
└──────────────────────────────────────────┘
```

---

## 📚 Key Concepts

### Stock Analysis
- **Technical Indicators**: Mathematical calculations on price/volume
  - RSI: Momentum indicator (0-100)
  - MACD: Trend-following indicator
  - Bollinger Bands: Volatility indicator

- **Signals**: BUY/SELL/HOLD recommendations based on multiple indicators
- **Predictions**: LSTM neural network forecasts (30 days)

### RL Trading
- **Agent**: AI that learns to trade (PPO or A2C algorithm)
- **Training**: Agent practices trading on historical data
- **Backtest**: Testing strategy on past data
- **Buy & Hold**: Simple baseline (buy and never sell)
- **Momentum**: Buy on uptrends, sell on downtrends

### Risk Metrics
- **Return**: Profit/loss percentage
- **Volatility**: How much prices fluctuate
- **Sharpe Ratio**: Return per unit of risk (>1 is good)
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

---

## ⚠️ Important Notes

### Educational Use Only
- This is for **learning and research**
- **NOT financial advice**
- **NOT for real trading decisions**
- Past performance ≠ future results

### Data Sources
- **Yahoo Finance**: Real-time and historical data
- **Free**: No API key required
- **Delays**: 15-minute delay on some markets

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB+ (for RL training)
- **Storage**: 500MB+ (for models and data)
- **Internet**: Required for stock data

---

## 🐛 Troubleshooting

### "Training failed" Error
**Cause**: Not enough historical data
**Fix**: Choose a stock with longer history (AAPL, MSFT, etc.)

### Charts Not Displaying
**Cause**: Missing Plotly dependency
**Fix**: `pip install plotly`

### Slow Training
**Cause**: Large number of timesteps
**Fix**: Reduce to 30,000 steps for testing

### "Module not found" Error
**Cause**: Dependencies not installed
**Fix**: `pip install -r requirements.txt`

---

## 🎓 Learn More

- **RL_Design.md**: Detailed RL system documentation
- **UI_IMPROVEMENTS.md**: UI design and architecture
- **README.md**: Project overview and features

---

## 🆘 Need Help?

1. Check **RL_Design.md** for RL questions
2. Check **UI_IMPROVEMENTS.md** for UI questions
3. Check logs in `data/logs/app.log`
4. Open an issue on GitHub

---

## 🎉 You're Ready!

Start by:
1. Clicking "AAPL" to see a quick analysis
2. Switching to RL Trading and running a backtest
3. Training your first agent with default settings

**Enjoy exploring AI-powered stock analysis!** 📈🤖
