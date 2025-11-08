# Stock Agent Pro - UX Documentation
**Current Implementation | Last Updated: November 2025**

---

## Overview

Stock Agent Pro is a professional finance platform built with Panel for analyzing stocks using AI/ML techniques. The platform features a clean **light theme** design with **wide horizontal layouts** to minimize scrolling and maximize data visibility.

---

## Design System

### Color Palette (Light Theme)

```python
# Background Colors
BG_PRIMARY = "#FFFFFF"        # Main background
BG_SECONDARY = "#F8F9FA"      # Cards and panels
BG_HOVER = "#E9ECEF"          # Hover states

# Border Colors
BORDER_SUBTLE = "#DEE2E6"     # Dividers
BORDER_FOCUS = "#ADB5BD"      # Active borders

# Text Colors
TEXT_PRIMARY = "#212529"      # Primary text
TEXT_SECONDARY = "#495057"    # Secondary text
TEXT_MUTED = "#6C757D"        # Tertiary text

# Semantic Colors
SUCCESS_GREEN = "#0F9D58"     # Positive values
DANGER_RED = "#DC3545"        # Negative values
ACCENT_PURPLE = "#7C3AED"     # Primary actions
ACCENT_CYAN = "#0891B2"       # Secondary actions
```

### Typography

- **Font Family**: System fonts (Inter, -apple-system, BlinkMacSystemFont)
- **Monospace**: For numbers and prices
- **Sizes**: 0.75rem (small), 0.875rem (base), 1.25rem (heading)

---

## Navigation Structure

### Top Navigation (Fixed Header)

```
┌────────────────────────────────────────────────────────┐
│ 📊 Stock Agent Pro                                     │
│                                                        │
│ Dashboard | Analysis | Trading | Watchlist | Models   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**Pages:**
1. **Dashboard** - Market overview, watchlist, quick actions
2. **Analysis** - Stock charts, technical analysis, LSTM predictions
3. **Trading** - RL agent training and backtesting
4. **Live Trade** - Real-time paper trading simulation
5. **Watchlist** - Stock tracking with multiple views
6. **Models** - LSTM and RL model registry

### Sidebar (Left Panel)

**Watchlist Panel** (240px width)
- Live stock prices with real-time updates
- Color-coded percentage changes
- Click symbol → load in Analysis view
- Symbols: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA

---

## Page Layouts

### 1. Dashboard

**Layout: Two-column (60% / 40%)**

```
┌──────────────────────┬─────────────────────┐
│ Markets Overview     │ Quick Actions       │
│ (4 major indices)    │ (Action buttons)    │
│                      │                     │
│ - S&P 500            │ - Train LSTM        │
│ - NASDAQ             │ - Backtest          │
│ - Dow Jones          │ - Compare           │
│ - Russell 2000       │ - Report            │
│                      │                     │
│ [Refresh button]     │                     │
└──────────────────────┴─────────────────────┘
```

**Features:**
- Real-time market indices with sparklines
- Live price updates every 5 seconds
- Quick action buttons for common tasks
- Compact card-based layout

---

### 2. Analysis Page

**Layout: Two-column (70% / 30%)**

```
┌─────────────────────────────────────────────────────────┐
│ [AAPL ▼] Apple Inc.  $270.37  +2.15 (+1.22%)          │
│ [Analyze Button] [Force Retrain LSTM Model checkbox]   │
└─────────────────────────────────────────────────────────┘
┌──────────────────────────┬──────────────────────────────┐
│                          │ Trading Signal               │
│                          │ BUY (60% confidence)         │
│   Primary Chart          │ Support: $244.61             │
│   (Candlestick + Volume) │                              │
│                          ├──────────────────────────────┤
│   900px height           │ 30-Day Prediction            │
│                          │ $275.19 (+1.78%)             │
│                          │ Confidence: 0.892            │
│                          │                              │
│                          ├──────────────────────────────┤
│                          │ AI Analysis                  │
│                          │ (Natural language summary)   │
│                          │                              │
└──────────────────────────┴──────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Technical Analysis Tabs                                 │
│ [Statistics] [Predictions]                              │
└─────────────────────────────────────────────────────────┘
```

**Components:**

**A. Stock Header**
- Symbol selector with autocomplete (AAPL, GOOGL, MSFT, etc.)
- Current price with color-coded change
- Analyze button to trigger AI analysis
- Force retrain checkbox for LSTM models

**B. Main Chart** (Plotly)
- Candlestick chart with volume
- Moving averages (MA-20, MA-50, MA-200)
- Bollinger Bands
- RSI, MACD indicators
- Interactive zoom and pan
- Professional color scheme

**C. Trading Signal Card**
- BUY/SELL/HOLD recommendation
- Confidence percentage
- Support and resistance levels
- Entry and exit suggestions

**D. 30-Day Prediction Card**
- LSTM ensemble prediction
- Predicted price and change
- Confidence interval
- Model metadata (training date, MAE)

**E. AI Analysis Card**
- Natural language market summary
- Technical analysis insights
- Volume and momentum analysis
- Risk assessment

**F. Technical Analysis Tabs**
- **Statistics**: Key metrics, price levels
- **Predictions**: Detailed LSTM forecast data

---

### 3. Trading Page

**Layout: Single column with compact sections**

```
┌─────────────────────────────────────────────────────────┐
│ ⚡ Reinforcement Learning Trading Lab                   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Configuration Panel                                     │
│                                                         │
│ Symbol: [AAPL ▼]  Algorithm: [◉ PPO  ○ A2C]          │
│ Use LSTM Features: [☑]                                 │
│ Training Period: [========] 365 days                   │
│ Training Steps: [========] 50,000                      │
│                                                         │
│ [🚀 Start Training]  [📊 Run Backtest]                │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Results Panel                                           │
│ (Training progress, backtest results)                   │
└─────────────────────────────────────────────────────────┘
```

**Features:**

**A. Configuration Panel**
- Symbol selection (8 major stocks)
- Agent type: PPO or A2C
- LSTM feature enhancement toggle
- Training period slider (30-730 days)
- Training steps slider (10k-100k)

**B. Training Monitor**
- Real-time progress bar
- Episode count and rewards
- Training time estimation
- Progress chart (episode rewards over time)

**C. Backtest Results**
- Performance comparison table (RL Agent vs Buy&Hold vs Momentum)
- Metrics: Return %, Sharpe Ratio, Max Drawdown, Win Rate
- Action distribution (SELL, HOLD, BUY_SMALL, BUY_LARGE)
- Portfolio value chart over time (for live trading sessions)
- Action comparison visualizations

---

### 4. Live Trade Page

**Layout: Configuration + Real-time Dashboard**

```
┌─────────────────────────────────────────────────────────┐
│ Configuration Panel                                     │
│ Symbol: [AAPL ▼] Algorithm: [PPO ▼]                   │
│ Initial Capital: [$10,000] Max Position: [100]         │
│ Stop Loss: [5%]                                        │
│ [▶ Start Trading] [⏸ Pause] [■ Stop]                  │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Trading Status           Live Trading Portfolio         │
│ Status: ACTIVE           Total: $10,523.45  +5.23%     │
│ Runtime: 2h 34m          Cash: $4,200.00    39.9%      │
│ Last Update: 10:45:23    Invested: $6,323.45  60.1%    │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Current Positions                                       │
│ AAPL: 25 shares @ $252.94 avg | Current: $254.12       │
│ Unrealized P&L: +$29.50 (+1.18%)                       │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Recent Trades                    Event Log              │
│ 10:43 BUY 5 @ $253.45           SESSION_START           │
│ 10:15 SELL 3 @ $251.20          TRADE: BUY 5            │
│ 09:52 BUY 10 @ $249.80          HOLD - No action       │
└─────────────────────────────────────────────────────────┘
```

**Features:**

**A. Configuration Panel**
- Symbol selection with autocomplete
- Algorithm dropdown (PPO/A2C) - auto-loads trained model
- Initial capital input ($10,000 default)
- Max position size (shares limit)
- Stop-loss percentage

**B. Trading Dashboard** (shown when active)
- Trading Status: Status, runtime, last update
- Live Trading Portfolio: Total value, cash, invested, P&L
- Current Positions: Holdings with unrealized P&L
- Recent Trades: Trade history with agent decisions
- Event Log: System events and notifications

**C. Controls**
- Start Trading: Begin live session
- Pause: Suspend trading (keep positions)
- Stop: End session and clear results

**D. Risk Management**
- Auto stop-loss on positions
- Position size limits
- Circuit breakers
- Market hours enforcement

**Important:**
- Paper trading only (no real money)
- Real-time Yahoo Finance data
- 60-second trading cycles
- Educational purpose only

---

### 5. Watchlist Page

**Purpose: Simple stock price tracker**

**Single Table View:**
- Compact table: Symbol, Price, Change, Volume, Market Cap
- Add symbols using input field
- Remove symbols with "×" button
- Real-time price updates

**Columns:**
- Symbol
- Current Price
- Daily Change ($ and %)
- Volume
- Market Cap

*Note: This is a price tracker only, not a portfolio manager. No position tracking (shares, cost basis, P&L). For portfolio management, use the Live Trade page.*

---

### 6. Models Page

**Layout: Tabbed interface with dynamic header**

```
┌─────────────────────────────────────────────────────────┐
│ LSTM Models                                             │
│ Trained prediction models                               │
├─────────────────────────────────────────────────────────┤
│ 🧠 LSTM Models │ 🤖 RL Agents                           │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Model   Symbol  Trained     Final   Val    Size    │ │
│ │                             Loss    Loss            │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ AAPL    AAPL    2025-01-01  0.0234  0.0256  3 models││
│ │ GOOGL   GOOGL   2025-01-01  0.0198  0.0223  3 models││
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

(Header changes to "RL Trading Agents / Reinforcement learning models"
 when switching to RL Agents tab)
```

**Dynamic Header:**
- Header updates based on selected tab
- LSTM tab: "LSTM Models / Trained prediction models"
- RL Agents tab: "RL Trading Agents / Reinforcement learning models"

**LSTM Models Tab:**
- Lists all trained LSTM ensemble models
- Shows training performance (Final Loss, Validation Loss)
- Displays number of models in ensemble
- Training date for each model

**RL Agents Tab:**
- Lists all trained RL agents (PPO, A2C)
- Shows agent type and symbol
- Training date
- Performance note: "Run backtest →" (metrics calculated on-demand)

*Note: RL performance data is generated during backtesting, not stored with models*

---

## Component Specifications

### Cards

All cards follow consistent styling:
- Background: `BG_SECONDARY` (#F8F9FA)
- Border: 1px solid `BORDER_SUBTLE` (#DEE2E6)
- Border-left accent: 4px solid accent color
- Border-radius: 8px
- Padding: 15px
- Box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08)

### Tables

**Styled tables with:**
- Light background (#f3f4f6)
- Alternating row colors
- Monospace font for numeric values
- Color-coded positive/negative values
- Hover effects

### Buttons

**Primary buttons:**
- Background: `ACCENT_PURPLE` (#7C3AED)
- White text
- Border-radius: 6px
- Hover effect with shadow

**Secondary buttons:**
- Background: `BG_SECONDARY`
- Border: 1px solid `BORDER_SUBTLE`
- Hover background: `BG_HOVER`

---

## Layout Philosophy

### Wide Horizontal Layouts

**Principle**: Minimize vertical scrolling by using wide horizontal layouts

**Implementation:**
1. **Dashboard**: Side-by-side panels (Markets + Quick Actions)
2. **Analysis**: 70/30 split (Chart + Signals/Predictions stacked)
3. **Trading**: Single column with compact sections
4. **Models**: Vertically stacked sections with horizontal tables

**Benefits:**
- More data visible at once
- Reduced scrolling on large screens
- Better use of widescreen monitors
- Professional desktop application feel

### Card Layout Fix

**Important**: Cards should stack vertically, not horizontally
- Trading Signal card (100% width)
- 30-Day Prediction card (100% width)
- Each card has margin-bottom for spacing
- No flexbox side-by-side layout for content cards

---

## Key Features Implemented

### ✅ Real-time Data
- Live market indices
- Real-time watchlist prices
- Auto-refresh capabilities

### ✅ LSTM Predictions
- 30-day price forecasts
- Ensemble model training (3 models)
- Confidence intervals
- Model performance tracking (Final Loss, Val Loss)

### ✅ RL Trading Agents
- PPO and A2C algorithms
- LSTM-enhanced feature option
- Live training progress
- Comprehensive backtesting
- Strategy comparison (RL vs Buy&Hold vs Momentum)

### ✅ Live Trading Simulation
- Paper trading with real-time data
- Trained agent execution (PPO/A2C)
- Real-time portfolio tracking
- Risk management controls
- Live monitoring dashboard

### ✅ Technical Analysis
- Multiple indicators (RSI, MACD, Bollinger Bands)
- Moving averages (20, 50, 200-day)
- Support/resistance levels
- Chart pattern recognition

### ✅ AI Analysis
- Natural language market summaries
- Trading recommendations (BUY/SELL/HOLD)
- Confidence scoring
- Risk assessment

---

## Data Flow

### Analysis Workflow

```
User selects symbol → Fetch historical data → Display chart
                    → Calculate indicators
                    → Load LSTM model (if exists)
                    → Generate predictions
                    → Run AI analysis (Ollama)
                    → Display signals and recommendations
```

### Training Workflow

```
Configure LSTM → Fetch data → Train ensemble (3 models)
                            → Save models and metadata
                            → Display training history

Configure RL   → Fetch data → Create trading environment
                            → Train agent (PPO/A2C)
                            → Save model checkpoints
                            → Show progress and results
```

### Backtest Workflow

```
Select symbol → Load trained RL agent → Run backtest
                                     → Compare with baselines
                                     → Calculate metrics
                                     → Visualize results
```

---

## File Structure

```
src/
├── ui/
│   ├── design_system.py       # Colors, HTML components, table styles
│   ├── __init__.py            # Module exports
│   └── pages/
│       ├── analysis.py        # Main app, watchlist sidebar, analysis page
│       ├── dashboard.py       # Market overview
│       ├── trading.py         # RL training UI
│       ├── live_trading.py    # Live paper trading simulation
│       ├── portfolio.py       # Watchlist (stock tracker)
│       └── models.py          # Model registry
├── tools/
│   ├── stock_fetcher.py       # Yahoo Finance API
│   ├── visualizer.py          # Chart generation
│   ├── technical_analysis.py  # Indicators (RSI, MACD, BB)
│   └── lstm/
│       ├── prediction_service.py  # LSTM predictions
│       ├── model_architecture.py  # Neural network models
│       └── data_pipeline.py       # Data preprocessing
├── rl/
│   ├── training.py            # RL training logic
│   ├── backtesting.py         # Backtest engine
│   ├── environments.py        # Trading environment
│   ├── live_trading.py        # Live trading engine
│   ├── session_manager.py     # Session persistence
│   └── visualizer.py          # RL charts
└── agents/
    ├── query_processor.py     # AI analysis (Ollama)
    └── hybrid_query_processor.py  # Ollama + regex fallback
```

---

## Recent Changes

### January 2025

1. **Watchlist Implementation**
   - Added live watchlist to sidebar
   - Real-time price updates with color-coded changes
   - 6 major stocks: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA

2. **Dashboard Cleanup**
   - Removed Featured Stocks section
   - Simplified to Markets + Quick Actions layout

3. **Analysis Page Fixes**
   - Removed Indicators data table
   - Fixed card layout: Trading Signal and Prediction cards now stack vertically
   - Changed from side-by-side to full-width stacked cards

4. **Models Page Enhancements**
   - Fixed LSTM model discovery (changed from subdirectories to metadata file pattern)
   - Added performance metrics: Final Loss, Val Loss
   - Updated RL models to show "Run backtest →" hint instead of N/A

5. **Code Organization**
   - Consolidated UI components into pages directory
   - All page implementations now in `src/ui/pages/`
   - Improved project structure consistency

---

## Known Limitations

1. **RL Model Performance**
   - Performance metrics (return %, Sharpe) not persisted to disk
   - Must run backtest to see performance data
   - Could be improved by saving backtest results with models

2. **Watchlist Page**
   - Simple table-based stock price tracker
   - Shows real-time prices, changes, volume, market cap
   - Note: Not a portfolio manager - no position/P&L tracking
   - For actual portfolio tracking, see Live Trading page

3. **Real-time Updates**
   - Uses polling (5-second intervals)
   - Could be improved with WebSocket for true real-time

4. **Model Management**
   - No model deletion UI
   - No model comparison features
   - Limited metadata displayed

---

## Future Enhancements

### Short-term
- [ ] Save RL backtest results as metadata
- [ ] Add model deletion functionality
- [ ] Implement model comparison view
- [ ] Complete portfolio page implementation

### Medium-term
- [ ] WebSocket for real-time prices
- [ ] News integration and sentiment analysis
- [ ] Multi-symbol comparison charts
- [ ] Advanced alert system

### Long-term
- [ ] Options and derivatives support
- [ ] Fundamental analysis integration
- [ ] Mobile app version
- [ ] API access for external tools

---

## Design Principles

1. **Clarity Over Cleverness** - Simple, clear interfaces
2. **Data Density** - Show maximum useful information
3. **Minimal Scrolling** - Wide horizontal layouts
4. **Professional Aesthetics** - Clean, light theme
5. **Fast Feedback** - Real-time updates, progress indicators
6. **Consistent Patterns** - Reuse UI components and layouts

---

## Contact

For questions or suggestions about the UX implementation, see:
- Design system: `src/ui/design_system.py`
- Main app: `src/ui/pages/analysis.py`
- Page implementations: `src/ui/pages/`

Last updated: November 2025
