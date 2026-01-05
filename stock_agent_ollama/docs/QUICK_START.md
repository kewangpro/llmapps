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
- **Interactive Watchlist** with clickable stock cards
- Real-time prices and daily changes (updates every 5 seconds)
- Shows live positions from active trading sessions (shares + value)
- Click any card → instantly navigate to Analysis page
- Automatically syncs with your portfolio
- Supports any stock ticker symbol, not just predefined ones

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
2. Enter or select a stock symbol (e.g., AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, TEAM, or any valid ticker)
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

### Training an Agent

1. Click **Trading** tab
2. **Configure Agent**:
   - **Symbol**: Enter any valid ticker (e.g., AAPL, TSLA, NVDA)
   - **Algorithm**: PPO (stable), RecurrentPPO (LSTM memory), or Ensemble (combined)
   - **Training Period**: 1095 days (3 years recommended)
   - **Training Steps**: 300,000 (recommended)
   - **Learning Rate**: Auto-set per algorithm
   - **Initial Balance**: $100,000
3. Click **"🚀 Start Training"**
4. Monitor progress bar and chart
5. Review training results when complete

**Training Results:**
- Summary with agent details, episodes, training time
- Diagnostics: Invalid action rate, episode reward, portfolio return
- Metrics: Win rate, episode rewards, explained variance
- Progress chart showing reward improvement
- Action distribution pie chart
- Auto-saved to `data/models/rl/ALGO_SYMBOL_timestamp/`
  - `best_model.zip`: Peak performance (used for backtesting)
  - `final_model.zip`: End-of-training

**Always Enabled:**
- Action Masking: Automatically prevents invalid trades during execution
- 6-Action Space: HOLD, BUY_SMALL/MEDIUM/LARGE, SELL_PARTIAL/ALL
- Invalid actions are replaced with HOLD for safe operation

### Running Backtests

1. **Trading** tab
2. Select stock symbol
3. Click **"📊 Run Backtest"**
4. Wait ~30 seconds
5. Review results

**Automatic Model Loading:**
- Automatically finds all trained models for selected symbol
- Compares PPO, RecurrentPPO, Ensemble (if trained)
- All models included automatically

**Backtest Results:**
- **Performance Table**: All RL agents vs Buy & Hold vs Momentum
  - Metrics: Total Return %, Sharpe Ratio, Max Drawdown, Win Rate
  - Action distribution across all 6 actions
- **Charts**:
  - Portfolio value comparison
  - Stock price with trade signals (buy/sell markers for each RL agent)
  - Action distribution visualization
  - Key metrics bar chart

---

## 🔴 Live Trading

### Starting & Resuming a Live Trading Session

The live trading session is **persistent**. You can stop the application and restart it, and your session will be automatically saved and reloaded.

**Starting a New Session:**
1. Click **Live Trade** tab
2. **Configure Settings**:
   - **Symbol**: Enter any valid ticker (e.g., AAPL, TSLA, NVDA)
   - **Algorithm**: PPO, RecurrentPPO, or Ensemble (auto-loads trained model)
   - **Initial Capital**: Starting balance ($100,000 default)
   - **Max Position %**: Maximum position (80% default)
   - **Stop Loss**: Auto stop-loss percentage (5% default)
   - **Auto Select Stock**: Enable to dynamically rotate between watchlist stocks
     - When enabled, Symbol and Algorithm inputs are disabled
     - System continuously evaluates rotation opportunities after cooldown
     - Automatically selects best performing stock (prioritizes backtest performance, falls back to 5-day price return)
     - Chooses optimal algorithm using dynamic scoring based on Sharpe ratio and returns
     - Idle detection: Applies -50% penalty after 20 idle cycles to force rotation from inactive stocks
     - Shadow mode: When >50% cash and idle for 3+ cycles, actively scans for live BUY signals
     - Recency penalty: Applies -30% penalty to stocks rotated away from within last 30 minutes
     - Automatically closes positions before rotation to capture better opportunities
     - Single AUTO session policy prevents duplicate sessions
     - Maximizes capital efficiency by trading strongest performers
3. Click **"Create & Start Session"**
4. Monitor real-time updates

**Resuming a Session:**
1. Start the application: `python src/main.py`
2. The previous session will be loaded automatically from `data/live_sessions/live_session.json`
3. Navigate to the **Live Trade** tab to view and manage your session
4. You can continue trading or stop the session

**Live Trading Dashboard:**
- **Trading Status**: Session status with AUTO badge for auto-select mode, runtime, last update
- **Portfolio Summary**: Total value, cash, invested, P&L
- **Session Stats**: Trade count with total fees displayed (e.g., "5 • $245.50")
- **Current Positions**: Holdings with unrealized P&L
- **Recent Trades**: Trade history with full timestamps, symbol, action, shares, price, **COST** (transaction costs), and P&L
- **Event Log**: System events with symbol prefixes for auto-select (e.g., `[HOOD] Agent predicted SELL_ALL...`), includes rotation events with model names (e.g., `Rotated to HOOD (recurrent_ppo_HOOD_20251203_103214)`)

**Transaction Costs (Realistic Trading Simulation):**
- **$0 commissions + 0.05% slippage** (zero-commission era)
- **Round-trip cost: 0.1%** (buy + sell)
- Reflects modern brokers (Fidelity, Schwab, Robinhood)
- Buy costs shown as negative P&L in trade record
- Sell costs subtracted from realized P&L
- All costs match backtesting environment exactly
- Full transparency in UI (displayed in trades table and session stats)

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
- Trading cycle runs every 3600 seconds (1 hour) default
- Requires trained RL model for the selected symbol (or watchlist stocks for auto-select mode)
- Sessions auto-save every 5 minutes and on stop
- Session state persists across application restarts
- Sessions displayed in chronological order (newest first) for easy access
- Session IDs: `SESSION_AUTO_*` for auto-select, `SESSION_SYMBOL_*` for manual
- Auto-select sessions display cyan "AUTO" badge next to symbol in session table
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
- Models sorted by training date (newest first)
- Click "View" to see details (if enabled)

**Tab 2: RL Agents**
- Header: "RL Trading Agents / Reinforcement learning models"
- Lists all trained PPO, RecurrentPPO, and Ensemble agents
- Checkbox selection for batch backtesting
- Shows algorithm type, symbol, training date
- Models sorted by training date (newest first)
- Backtest button appears when models are selected
- Performance column shows "Run backtest →" hint
  - Performance calculated when you run backtests
  - Not stored with models

**Batch Backtesting:**
- Check boxes next to models you want to backtest
- Backtest button dynamically shows count of selected models
- Run comprehensive backtests on multiple models simultaneously
- View unified performance comparison across all selected models

**Features:**
- Header dynamically updates when switching between tabs
- Clean tabbed interface for better organization
- Auto-loads models on page visit
- Chronological ordering (newest first) for easy access to recent models

---

## 📋 Watchlist Page

**Purpose**: Simple stock price tracker with smart suggestions

**Features:**
- **Top Movers Suggestions**: Automatically finds and displays 8 high-momentum stocks (30-day max returns) from curated universe
  - Data cached for 5 minutes for optimal performance
  - Manual refresh button (🔄 Refresh) to force fresh market data on demand
  - Automatically updates on browser refresh
- **Real-time Tracking**: Monitor current prices, daily changes, and volume
- **Compact Table**: View key metrics (Price, Change, 52W Range, Market Cap)
- **Easy Management**: Add/remove symbols with a single click

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
- ✅ **3 Algorithms**: PPO, RecurrentPPO, Ensemble
- ✅ **RecurrentPPO** uses LSTM memory with trend indicators
- ✅ **Ensemble** combines PPO (aggressive) + RecurrentPPO (risk-managed) with weighted voting
- ✅ **PPO** stable baseline for general trading
- ✅ **Advanced Risk Management** (5% stop-loss, 3% trailing stop, 15% circuit breaker)
- ✅ **Market Regime Detection** (BULL, BEAR, SIDEWAYS, VOLATILE)
- ✅ **Multi-Timeframe Features** (weekly/monthly trend analysis)
- ✅ **Kelly Position Sizing** (optimal sizing based on edge)
- ✅ **Ensemble Agents** (combine multiple algorithms)
- ✅ **Action Masking** prevents invalid trades
- ✅ **6-Action Space** fine-grained position sizing
- ✅ **Algorithm-Specific Rewards** optimized per algorithm
- ✅ **1095 days** (3 years) optimal for diverse conditions
- ✅ **300,000 steps** recommended (15-35 min training)
- ✅ **Backtesting** auto-loads all models
- ✅ Training runs in background
- ✅ Auto-saves best model

### Performance
- ⚡ Real-time data with 5-second UI refresh
- ⚡ Intelligent multi-tier caching system for optimal performance:
  - Real-time quotes: 1 minute cache for live price data
  - Bulk data/Top Movers: 5 minutes cache for high-frequency requests
  - Company fundamentals: 1 hour cache for stable information
  - Historical OHLCV: 1 day cache for long-term charts
  - Automatic cache invalidation on manual refresh
- ⚡ Manual refresh button available for Top Movers to force fresh market data
- ⚡ Reduce training steps to 100k for quick experiments (300k recommended)
- ⚡ LSTM predictions load existing models or auto-train new ones

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

### Workflow 3: Train RL Agent
```
1. Click Trading tab
2. Select symbol: NVDA
3. Choose algorithm: PPO, RecurrentPPO, or Ensemble
4. Set training period: 1095 days (3 years)
5. Set timesteps: 300,000 (recommended)
6. Click "🚀 Start Training"
7. Monitor progress
8. Review results
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

### Workflow 5: Live Trading Session
```
1. Click Live Trade tab
2. Select symbol: AAPL
3. Choose algorithm: PPO, RecurrentPPO, or Ensemble
4. Set initial capital: $100,000
5. Click "Create & Start Session"
6. Monitor portfolio, positions, trades
7. Click "Stop" when done
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

**PPO (Proximal Policy Optimization)**
- Stable on-policy baseline
- Clipped objective for training stability
- Good for: General-purpose trading

**RecurrentPPO**
- LSTM memory for temporal patterns
- Trend indicators (SMA_Trend, EMA_Crossover, Price_Momentum)
- Enhanced reward for trend-following
- Good for: Markets with temporal dependencies

**Ensemble (PPO + RecurrentPPO)**
- Weighted voting system combining both algorithms
- PPO (30%): Opportunistic growth strategy
- RecurrentPPO (70%): Primary strategy with LSTM memory and trend-following
- Confidence-based decisions with RecurrentPPO as primary
- Good for: Balanced performance with superior risk-adjusted returns

**Recommendation**: Train all algorithms and compare via backtesting. All 3 algorithms (PPO, RecurrentPPO, Ensemble) are reliable and well-tuned.

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
- Default is 300,000 steps for best results
- Can reduce to 100,000 steps for quick testing
- Reduce training period to 365 days for faster experiments
- Close other applications to free up resources

**Expected Training Times (300k steps):**
- PPO: ~15-20 minutes (efficient on-policy)
- RecurrentPPO: ~25-35 minutes (LSTM requires more compute)
- Ensemble: ~40-55 minutes (trains both PPO + RecurrentPPO)

### Backtest Validation

After running backtests, you can validate the results for mathematical correctness:

```bash
# Validate all algorithms for a symbol
python validate_backtest.py --symbol AAPL

# Validate all watchlist stocks (all algorithms)
python validate_backtest.py --watchlist

# Validate specific algorithm only
python validate_backtest.py --symbol AAPL --algorithm ppo
python validate_backtest.py --watchlist --algorithm ensemble
```

**What it checks:**
- Return calculation accuracy
- Action distribution sums to 100%
- Win rate calculation correctness
- Portfolio value consistency
- Metrics reasonableness (Sharpe ratio <8.0, win rates based on sample size)
- Transaction cost inclusion (0.05% per trade, slippage only)
- Reproducibility hints

**Expected output:**
```
✅ PASS Return calculation
✅ PASS Action distribution sums to 100%
✅ PASS Win rate calculation
✅ PASS Transaction costs are applied
Overall: 8/8 checks passed (100.0%)
```

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

## 🧪 Testing

### Running Tests

The project includes a comprehensive test suite with 48 automated tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_action_masking.py -v

# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html
# View at: htmlcov/index.html
```

**Test Coverage:**
- Configuration management (96% coverage)
- Action masking logic
- Live trading data models
- RL components (ensemble, environments)
- Technical analysis indicators

**Test Files:**
- `tests/test_config.py` - Configuration validation (14 tests)
- `tests/test_action_masking.py` - Action masking (10 tests)
- `tests/test_live_trading_models.py` - Data models (14 tests)
- `tests/test_rl_components.py` - RL components (5 tests)
- `tests/test_technical_analysis.py` - Technical indicators (5 tests)

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
- **Live Sessions**: `data/live_sessions/SESSION_*.json` (multi-session support)

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

## 🔧 Key Features

### Algorithm-Specific Optimizations
- **RecurrentPPO**: LSTM memory with trend indicators
  - SMA_Trend, EMA_Crossover, Price_Momentum auto-enabled
  - `RecurrentPPORewardConfig`: Hold winner and momentum bonuses
  - 13-feature observation (10 base + 3 trend)
- **PPO**: Stable on-policy baseline
  - `PPORewardConfig`: Strong penalties prevent action collapse
- **Ensemble**: Combined strategy
  - Uses both `PPORewardConfig` and `RecurrentPPORewardConfig`
  - Weighted voting balances aggressive and risk-managed approaches

### Configuration System
- **Single Source of Truth**: Centralized in `env_factory.py`
- **Consistent Defaults**: Shared across training/backtesting/live trading
- **No Train-Test Mismatch**: Auto-loads exact training config
- **Conditional Features**: Trend indicators for RecurrentPPO only
- **Advanced Improvements**: Risk management, regime detection, multi-timeframe features, Kelly sizing enabled by default

### Bug Fixes
- **Short-Selling Prevention**: Fixed accidental short positions
- **Symbol Input**: Accepts any valid ticker
- **Position Limits**: Consistent 80% across systems
- **Floating Point Precision**: Tolerance prevents rejection

### Live Trading Enhancements
- **Multi-Session Support**: Run multiple strategies
- **Session Persistence**: Auto-save and resume
- **Improved UI**: Better session management
- **Config Loading**: Matches training environment exactly

---

**Ready to explore AI-powered finance!** 📈🤖

*Educational platform only - Not for real trading decisions*