# Stock Agent Pro - Quick Start Guide

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

## 🎯 Interface Overview

### Main Navigation

The platform features a professional light-theme interface with 6 main pages:

```
┌─────────────────────────────────────────────────────────────────┐
│ 📊 Stock Agent Pro                                              │
├─────────────────────────────────────────────────────────────────┤
│ Dashboard | Analysis | Trading | Live Trade | Watchlist | Models│
└─────────────────────────────────────────────────────────────────┘
```

**Page Overview:**
- **Dashboard** - Market overview and quick actions
- **Analysis** - Stock charts, technical analysis, LSTM predictions
- **Trading** - RL agent training and backtesting
- **Live Trade** - Real-time paper trading with trained agents
- **Watchlist** - Track stocks with multiple view options
- **Models** - LSTM and RL model registry

**Sidebar:**
- Live **Watchlist** with real-time prices
- Stocks: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA
- Click any symbol to analyze

---

## 📊 Dashboard

### Quick Overview (1 click)
1. Click **Dashboard** tab
2. View market indices (S&P 500, NASDAQ, Dow Jones, Russell 2000)
3. Monitor your watchlist stocks
4. Use quick actions for common tasks

**Features:**
- Real-time market data with live updates
- Quick action buttons (Train LSTM, Backtest, Compare, Report)
- Compact card-based layout

---

## 📈 Stock Analysis

### Quick Analysis
1. Click **Analysis** tab
2. Select a stock from the dropdown (AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, TEAM)
3. Optional: Check "Force Retrain LSTM Model" to retrain
4. Click **Analyze** button
5. View comprehensive results

### What You'll See

**Left Panel (70%):**
- **Stock Header**: Company name, current price, daily change
- **Interactive Chart**: Candlestick chart with volume
  - Moving averages (MA-20, MA-50, MA-200)
  - Bollinger Bands
  - RSI and MACD indicators
  - Zoom and pan capabilities
- **Technical Analysis Tabs**:
  - Statistics: Key metrics and price levels
  - Predictions: Detailed LSTM forecast data

**Right Panel (30%):**
- **Trading Signal**: BUY/SELL/HOLD recommendation
  - Confidence percentage
  - Support/resistance levels
  - Entry and exit suggestions
- **30-Day Prediction**: LSTM forecast
  - Predicted price and change percentage
  - Confidence interval
  - Model metadata (training date, accuracy)
- **AI Analysis**: Natural language market summary
  - Technical trends
  - Volume and momentum analysis
  - Risk assessment

---

## 🤖 RL Trading

### Training an Agent (5-10 minutes)

1. Click **Trading** tab
2. **Configure Agent**:
   - **Symbol**: Select from dropdown (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, ORCL)
   - **Algorithm**: Choose DQN (best performance), PPO (stable), or A2C (faster)
   - **Use LSTM Policy**: Optional for PPO only (RecurrentPPO for bear markets)
   - **Training Period**: 1095 days (3 years, proven optimal)
   - **Training Steps**: 100,000 (recommended starting point)
   - **Learning Rate**: Auto-set based on algorithm
   - **Initial Balance**: $100,000
3. Click **"🚀 Start Training"**
4. Monitor progress bar and real-time chart
5. Review training results when complete

**Training Results:**
- Summary card with agent details, episodes, and training time
- Training diagnostics: Invalid action rate, mean episode reward, portfolio return
- Training metrics: Win rate, final/best episode reward, explained variance
- Training progress chart showing reward improvement over episodes
- Action distribution pie chart showing how agent used each action
- Model automatically saved to `data/models/rl/ppo_SYMBOL_timestamp/`
  - `best_model.zip`: Peak performance model (used for backtesting)
  - `final_model.zip`: End-of-training model

**Always Enabled:**
- Action Masking: Prevents invalid trades (e.g., selling with no shares)
- 6-Action Space: HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_PARTIAL, SELL_ALL

### Running Backtests

1. **Trading** tab
2. Select stock symbol from dropdown
3. Click **"📊 Run Backtest"**
4. Wait ~30 seconds
5. Review comprehensive results

**Automatic Model Loading:**
- Backtesting automatically finds and loads ALL available trained models for the selected symbol
- Compares PPO, A2C, DQN, and LSTM PPO agents (if trained)
- No need to select algorithm - all models are included

**Backtest Results:**
- **Performance Table**: All RL Agents vs Buy & Hold vs Momentum
  - Metrics: Total Return %, Sharpe Ratio, Max Drawdown, Win Rate
  - Action distribution (HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_PARTIAL, SELL_ALL)
- **Charts**:
  - Clean portfolio value comparison (all strategies)
  - Action distribution visualization
  - Key metrics bar chart

---

## 🔴 Live Trading

### Starting & Resuming a Live Trading Session

The live trading session is **persistent**. You can stop the application and restart it, and your session will be automatically saved and reloaded.

**Starting a New Session:**
1. Click **Live Trade** tab
2. **Configure Settings**:
   - **Symbol**: Select stock (AAPL, MSFT, GOOGL, etc.)
   - **Algorithm**: Choose PPO, A2C, or DQN (auto-loads trained model)
   - **Initial Capital**: Starting balance ($100,000 default)
   - **Max Position %**: Maximum position as % of portfolio (40% default)
   - **Stop Loss**: Automatic stop-loss percentage (5% default)
3. Click **"Create & Start Session"**
4. Monitor real-time updates

**Resuming a Session:**
1. Start the application: `python src/main.py`
2. The previous session will be loaded automatically from `data/live_sessions/live_session.json`
3. Navigate to the **Live Trade** tab to view and manage your session
4. You can continue trading or stop the session

**Live Trading Dashboard:**
- **Trading Status**: Session status, runtime, last update
- **Portfolio Summary**: Total value, cash, invested, P&L
- **Current Positions**: Holdings with unrealized P&L
- **Recent Trades**: Trade history with agent decisions
- **Event Log**: System events and notifications

**Controls:**
- **Pause**: Suspend trading (keep positions open)
- **Stop**: End session and clear positions

**Risk Management:**
- Automatic stop-loss on positions
- Position size limits
- Circuit breakers for large losses
- Market hours enforcement (optional)

**Important Notes:**
- ⚠️ **Paper trading only** - No real money involved
- Uses real-time Yahoo Finance data (1-minute delayed)
- Trading cycle runs every 60 seconds
- Requires trained RL model for the selected symbol
- Sessions auto-save every 5 minutes and on stop
- Session state persists across application restarts
- Educational purpose only

---

## 🗂️ Models Page

### Viewing Trained Models

1. Click **Models** tab
2. View models organized in two tabs with dynamic header

**Tab 1: LSTM Models**
- Header: "LSTM Models / Trained prediction models"
- Lists all LSTM ensemble models (3 models per symbol)
- Shows performance metrics: Final Loss, Validation Loss
- Training date and model size
- Click "View" to see details (if enabled)

**Tab 2: RL Agents**
- Header: "RL Trading Agents / Reinforcement learning models"
- Lists all trained PPO, A2C, DQN, and LSTM PPO agents
- Shows algorithm type, symbol, training date
- LSTM PPO models clearly labeled with "lstm_ppo" prefix
- Performance column shows "Run backtest →" hint
  - Performance data calculated when you run backtests
  - Not stored with models
- Click "Load" to use in backtesting (if enabled)

**Features:**
- Header dynamically updates when switching between tabs
- Clean tabbed interface for better organization
- Auto-loads models on page visit

---

## 📋 Watchlist Page

**Purpose**: Simple stock price tracker

**Single Table View:**
- Compact table with Symbol, Price, Change, 52W Range, Volume, Market Cap
- Add symbols using the input field at the top
- Remove symbols with the "×" button
- Real-time price updates

**What it shows:**
- Current stock prices
- Daily price changes ($ and %)
- 52-week high/low range
- Trading volume
- Market capitalization

**Important:** This is a price tracker/watchlist only. It does NOT track:
- ❌ Shares owned
- ❌ Cost basis / entry price
- ❌ Position P&L
- ❌ Portfolio value

For actual portfolio tracking with positions and P&L, use the **Live Trade** page.

---

## 💡 Tips & Best Practices

### Stock Analysis
- ✅ Use sidebar watchlist for quick symbol switching
- ✅ Charts are interactive - hover for exact values
- ✅ Green signals = bullish, Red = bearish, Orange = neutral
- ✅ LSTM predictions require trained models (auto-train on first analysis)
- ✅ Force retrain checkbox updates models with latest data

### RL Trading
- ✅ **DQN** recommended for production (most consistent, adapts to markets)
- ✅ **A2C** excellent for bull markets (40.92% on GOOGL)
- ✅ **RecurrentPPO (LSTM)** best for bear markets (+1.36% on TEAM)
- ✅ **PPO** stable baseline (LSTM optional for temporal patterns)
- ✅ **Action Masking** always enabled - prevents invalid trades automatically
- ✅ **6-Action Space** provides fine-grained control over position sizing
- ✅ **Algorithm-Specific Rewards** - DQN and PPO/A2C use different optimized configs
- ✅ **1095 days** (3 years) proven optimal for diverse market conditions
- ✅ **100,000 steps** recommended starting point (5-10 min training)
- ✅ **Backtesting** automatically loads all trained models for comparison
- ✅ Training runs in background - UI stays responsive
- ✅ Models saved automatically with best model selection

### Performance
- ⚡ Real-time data with 5-second refresh
- ⚡ Stock data cached for faster queries
- ⚡ Reduce training steps to 30k for quick experiments
- ⚡ LSTM predictions load existing models or train new ones

---

## 🎯 Example Workflows

### Workflow 1: Quick Market Check (10 seconds)
```
1. Open Dashboard tab
2. Check market indices (S&P 500, NASDAQ, etc.)
3. Review watchlist prices and changes
4. Use quick actions if needed
```

### Workflow 2: Deep Stock Analysis (30 seconds)
```
1. Click Analysis tab
2. Select symbol from dropdown (e.g., AAPL)
3. Click Analyze button
4. Review chart with technical indicators
5. Check trading signals and AI analysis
6. Review 30-day LSTM prediction
```

### Workflow 3: Train RL Agent (5-10 minutes)
```
1. Click Trading tab
2. Select symbol: NVDA
3. Choose algorithm: DQN (or PPO, A2C)
4. Optional: Check "Use LSTM Policy" if PPO and downtrend expected
5. Use recommended settings: 1095 days (3 years), 100,000 steps
6. Click "🚀 Start Training"
7. Monitor progress (5-10 minutes)
8. Review training results and charts
```

### Workflow 4: Backtest Strategy (30 seconds)
```
1. Trading tab
2. Select symbol: AAPL
3. Click "📊 Run Backtest"
4. All trained models automatically loaded and compared
5. Review all RL agents vs Buy & Hold vs Momentum
6. Compare metrics and performance charts
7. Analyze action distribution across all strategies
```

### Workflow 5: Live Trading Session (Real-time)
```
1. Click Live Trade tab
2. Select symbol: AAPL
3. Choose algorithm: DQN (or PPO, A2C)
4. Set initial capital: $100,000
5. Click "Create & Start Session"
6. Monitor portfolio, positions, trades in real-time
7. Click "Stop" button when done
```

### Workflow 6: Model Management (5 seconds)
```
1. Click Models tab
2. Review all trained LSTM models
3. Check RL agents and training dates
4. Note: Run backtests to see RL performance
```

---

## 🔍 Understanding Performance Metrics

### LSTM Model Metrics
- **Final Loss**: Average loss across ensemble models (lower is better)
- **Val Loss**: Validation loss (measures overfitting, lower is better)
- **Size**: Number of models in ensemble (typically 3)

### RL Training Metrics
- **Win Rate**: Percentage of profitable episodes during training (>50% is good)
- **Final Episode Reward**: Last episode's reward (shows final performance)
- **Best Episode Reward**: Peak reward achieved (shows learning potential)
- **Explained Variance**: Model quality (0-1 scale, >0.7 excellent, measures how well value function predicts returns)
- **Action Distribution**: Percentage breakdown of actions taken (shows strategy diversity)
- **Invalid Action Rate**: Percentage of masked invalid actions (lower is better, <5% excellent)

### RL Backtest Metrics
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return (>1 good, >2 excellent)
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: Return divided by max drawdown
- **Max Drawdown**: Worst peak-to-trough decline (risk measure)
- **Win Rate**: Percentage of profitable trades

### Trading Actions (6-Action Space)
- **HOLD**: Do nothing (default, safe action)
- **BUY_SMALL**: Buy with ~15% of available cash (conservative)
- **BUY_MEDIUM**: Buy with ~30% of available cash (moderate)
- **BUY_LARGE**: Buy with ~50% of available cash (aggressive)
- **SELL_PARTIAL**: Sell 50% of current position (take partial profits)
- **SELL_ALL**: Sell entire position (full exit)

---

## 📚 Key Concepts

### Technical Analysis
- **RSI** (Relative Strength Index): 0-100 scale
  - <30: Oversold (potential buy)
  - >70: Overbought (potential sell)
- **MACD**: Trend strength and direction
- **Bollinger Bands**: Volatility and price extremes
- **Moving Averages**: Trend identification (20, 50, 200-day)

### LSTM Predictions
- Neural network trained on 2 years of historical data
- Ensemble of 3 models for robustness
- 30-day price forecasts with confidence intervals
- Auto-training on first analysis

### Reinforcement Learning
- **Agent**: AI that learns optimal trading policy
- **Environment**: Simulates realistic trading with costs
- **Reward**: Maximizes risk-adjusted returns
- **Training**: 1000s of episodes on historical data

### Algorithm Selection Guide

**For Production Trading:**
- Use **DQN** - Most consistent across all market conditions
  - Adapts strategy to uptrends and downtrends
  - Good risk management
  - 35.40% in bulls, -3.46% in bears (best loss control)

**For Bull Markets Only:**
- Use **A2C** - Excellent performance but no adaptation
  - 40.92% on GOOGL (only 0.70% behind Buy & Hold)
  - Will fail catastrophically in bear markets (-14.59%)
  - Only if you're confident in sustained uptrend

**For Bear Markets:**
- Use **RecurrentPPO** (PPO with LSTM enabled)
  - Only algorithm with positive returns in downtrends (+1.36% on TEAM)
  - Uses temporal patterns to detect trend reversals
  - Too conservative for bull markets (17.61% on GOOGL)

**For General Use:**
- Use **DQN** or train all three and let backtesting compare

---

## ⚠️ Important Notes

### Educational Use Only
- ✋ Platform for **learning and research**
- ✋ **NOT financial advice**
- ✋ **NOT for real trading decisions**
- ✋ Past performance ≠ future results
- ✋ Always consult qualified financial professionals

### Data & Requirements
- **Data Source**: Yahoo Finance (real-time, free, no API key)
- **Python**: 3.9 or higher required
- **RAM**: 8GB+ recommended for RL training
- **Storage**: 1GB+ for models and cache
- **Internet**: Required for stock data

---

## 🐛 Troubleshooting

### "Training failed" or "Insufficient data"
**Fix:**
- Choose established stocks (AAPL, MSFT, GOOGL)
- Reduce training period to 180 days
- Check internet connection

### "No LSTM models" on Analysis
**Fix:**
- Run analysis once - auto-trains model (5-10 min)
- Check `data/models/lstm/` directory
- Ensure sufficient disk space

### Charts Not Displaying
**Fix:**
- Install Plotly: `pip install plotly`
- Try Chrome browser
- Clear browser cache

### Slow RL Training
**Fix:**
- Start with 100,000 steps for quick testing
- Reduce training period to 365 days
- Close other applications
- Use PPO (more efficient than A2C)

**Expected Training Times:**
- 100k steps: ~5-8 minutes (quick testing)
- 200k steps: ~10-15 minutes (decent results)
- 300k steps: ~15-20 minutes (proven optimal, recommended)
- 400k+ steps: ~25-35 minutes (diminishing returns)
- Volatile stocks train slower due to curriculum learning stages

### "Module not found" Error
**Fix:**
- Activate environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.9+)

### RL Models Show "Run backtest →"
**Explanation:**
- RL performance calculated during backtesting
- Not stored with models
- Click "Run Backtest" in Trading tab to see metrics

---

## 🎓 Learn More

### Documentation
- **README.md**: Project overview and architecture
- **UX.md**: Interface design and layout philosophy
- **RL_DESIGN.md**: Detailed RL system design

### Key Directories
- **Models**: `data/models/lstm/` and `data/models/rl/`
- **Cache**: `data/cache/stock_data/`
- **Logs**: `data/logs/app.log`
- **Live Sessions**: `data/live_sessions/live_session.json`

### Debugging
1. Check logs: `tail -f data/logs/app.log`
2. Verify dependencies: `pip list | grep -E "panel|tensorflow|stable-baselines3"`
3. Browser console: F12 for JavaScript errors

---

## 🆘 Need Help?

### First Steps
1. Review this guide thoroughly
2. Check error messages in application
3. Look for similar issues in logs
4. Try the troubleshooting section above

### Resources
- **Interface Questions**: See UX.md for design details
- **RL Questions**: See RL_DESIGN.md for architecture
- **General Issues**: Check README.md

---

## 🎉 Quick Start Checklist

**For Beginners:**
- [ ] Open Dashboard, check market indices
- [ ] Click sidebar watchlist symbol (e.g., AAPL)
- [ ] Review Analysis page results
- [ ] Switch to Trading tab, run a backtest
- [ ] Explore Models page to see trained models

**For Advanced Users:**
- [ ] Train first RL agent (PPO, 365 days, 50k steps)
- [ ] Run backtest to compare strategies
- [ ] Start live trading session with trained agent
- [ ] Experiment with LSTM features enabled
- [ ] Compare different symbols and algorithms
- [ ] Analyze action distributions and metrics

---

**Ready to explore AI-powered finance!** 📈🤖

*Educational platform only - Not for real trading decisions*
