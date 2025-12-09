# Stock Agent Pro - UX Documentation
**Current Implementation**

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

## Application Architecture

### Overall Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Stock Agent Pro                        [Header]          │
├──────────────┬──────────────────────────────────────────────┤
│              │  Tab Navigation                              │
│   Sidebar    │  Dashboard | Analysis | Training |           │
│  (Watchlist) │  Live Trade | Watchlist | Models             │
│              ├──────────────────────────────────────────────┤
│   240px      │                                              │
│   width      │          Main Content Area                   │
│              │        (Active Page Content)                 │
│              │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

### Top Navigation (Tab Bar)

**Pages:**
1. **📊 Dashboard** - Market overview with indices and quick actions
2. **📈 Analysis** - Stock charts, technical analysis, LSTM predictions
3. **🤖 Training** - RL agent training and backtesting
4. **🔴 Live Trade** - Real-time paper trading simulation
5. **📋 Watchlist** - Stock tracking table with real-time prices
6. **🧠 Models** - LSTM and RL model registry

### Sidebar (Left Panel)

**Interactive Watchlist Panel** (240px width)
- Clickable stock cards with hover effects
- Live stock prices with real-time updates (every 5 seconds)
- Color-coded percentage changes (green▲/red▼)
- Active position tracking from live trading sessions
  - Shows shares owned and current value in purple
  - Aggregates positions across all active sessions
- Click any card → instantly navigate to Analysis page and analyze that stock
- Automatically syncs with portfolio manager
- Clean card-based design with borders and shadows

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
- Symbol input with autocomplete suggestions (AAPL, GOOGL, MSFT, etc.)
- Accepts any valid stock ticker symbol (not restricted to predefined list)
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
│ Symbol: [AAPL ▼]  Algorithm: [PPO ▼]                  │
│ Algorithms: PPO, RecurrentPPO, DQN, QRDQN              │
│ Training Period: [========] 1095 days                  │
│ Training Steps: [========] 300,000                     │
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
- Symbol input with autocomplete (accepts any valid ticker)
- Algorithm: PPO, RecurrentPPO, DQN, or QRDQN
- Training period slider (180-1095 days)
- Training steps slider (50k-500k, default 300k)

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

**Layout: Multi-Session Dashboard + Active Session Details**

```
┌─────────────────────────────────────────────────────────────────┐
│ Create New Session                                              │
│ Symbol: [AAPL ▼] Algorithm: [PPO ▼]                           │
│ Capital: [$100,000] Max Pos: [80%] Stop Loss: [5%]            │
│ [☐ Auto Select Stock (rotate stocks for max returns)]         │
│ [Create & Start Session]                                       │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Dashboard Card (Purple Gradient Header)                         │
│ Total Sessions: 3    Running: 2                                │
│ Total Portfolio Value: $350,278.57                             │
│ Aggregate P&L: $+278.57 (+0.08%)                               │
├─────────────────────────────────────────────────────────────────┤
│ Sessions Table                                                  │
│ Session ID           Symbol      Model Name      Status Actions│
│ SESSION_AUTO_...     NVDA AUTO   ens_NVDA_...    running [View][Stop]│
│ SESSION_AAPL_...     AAPL        ppo_AAPL_...    running [View][Stop]│
│ SESSION_GOOGL_...    GOOGL       ppo_GOOGL_...   stopped [View][Start]│
└─────────────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────┐
│ Active Session Details (Selected from table above)             │
├──────────────────────────┬──────────────────────────────────────┤
│ Session Status           │ Portfolio                            │
│ ● RUNNING                │ Total: $100,523.45  ▲ $523.45       │
│ Symbol: AAPL             │ Cash: $50,200.00                     │
│ Session ID: SESSION_...  │ Trades: 12                           │
│ Running Time: 02:34:15   │ P&L: +5.23% ($523.45)                │
├──────────────────────────┴──────────────────────────────────────┤
│ Positions                                                       │
│ SYMBOL   SHARES   AVG ENTRY   CURRENT   UNREALIZED P&L         │
│ AAPL     100      $268.49     $270.68   ▲ $218.61 (+0.81%)     │
├─────────────────────────────────────────────────────────────────┤
│ Recent Trades                 │ Event Log                       │
│ TIME     ACTION   SHARES PRICE│ 09:43:42 [SESSION_START]       │
│ 09:43:42 BUY_LARGE  50  $268  │ 09:43:42 [TRADE] BUY 50        │
│ 09:42:42 BUY_LARGE  50  $269  │ 09:42:42 [TRADE] BUY 50        │
│                               │ 09:44:42 [HOLD] No action      │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**

**A. Session Creation Panel**
- Symbol input with autocomplete (accepts any valid ticker, disabled when auto-select enabled)
- Algorithm selection (PPO/RecurrentPPO/Ensemble, disabled when auto-select enabled)
- Initial capital input ($10,000-$1,000,000)
- Max position % (percentage of portfolio, 5-100%, default 80%)
- Stop-loss percentage (1-20%)
- Auto Select Stock checkbox - enables dynamic stock rotation
  - When checked: Symbol and Algorithm inputs become disabled
  - System automatically selects best performing watchlist stocks
  - Creates session ID: SESSION_AUTO_YYYYMMDD_HHMMSS
- Manual mode creates session ID: SESSION_SYMBOL_YYYYMMDD_HHMMSS

**B. Dashboard Card** (Multi-Session Overview)
- **Aggregate Metrics** (purple gradient header):
  - Total Sessions count
  - Running Sessions count
  - Combined portfolio value across all sessions
  - Aggregate P&L (sum of all sessions)
- **Sessions Table**:
  - Session ID, Symbol (with cyan "AUTO" badge for auto-select sessions), Model name
  - Status (running/stopped/paused)
  - Sessions sorted by creation time (newest first)
  - View/Start/Stop buttons per session
- **Updates every 5 seconds** with live prices
- **Smart updates** on button clicks (no page refresh/scroll)

**C. Active Session Details** (Below Dashboard)
- **Session Status Card**: Status, symbol with AUTO badge for auto-select mode, runtime, session ID
- **Portfolio Card**: Total value, cash, trades count, P&L with live updates
- **Positions Table**: Current holdings with real-time prices and unrealized P&L
- **Recent Trades**: Last 10 trades with timestamps, actions, prices, P&L
- **Event Log**: Last 15 events with symbol prefixes for auto-select sessions, rotation events include model names (SESSION_START, TRADE, HOLD, ORDER_REJECTED, STOCK_ROTATION, SESSION_END)

**D. Real-Time Updates**
- Position prices update every 5 seconds with current market data
- Portfolio value recalculates automatically
- Dashboard metrics refresh across all sessions
- No page flickering or scroll jumps on updates
- Smooth in-place card updates

**E. Session Management**
- **Create**: New sessions with timestamped IDs
- **View**: Switch between sessions (updates details section only)
- **Start**: Resume stopped sessions (adds SESSION_RESUMED event)
- **Stop**: Halt active sessions (adds SESSION_END event)
- **Persistence**: Sessions auto-save every 60 seconds to JSON files
- **Resume**: Reload sessions after app restart

**F. Risk Management**
- Auto stop-loss on positions
- Max position size enforcement
- Daily loss circuit breaker (10% max)
- Max portfolio risk per trade (2%)
- Market hours enforcement (optional)
- Order validation before execution

**Important:**
- Paper trading only (no real money)
- Real-time Yahoo Finance data
- 60-second trading cycles
- Multi-session support (run multiple strategies)
- Session persistence across app restarts
- Educational purpose only

---

### 5. Watchlist Page

**Purpose: Simple stock price tracker**

**Single Table View:**
```
┌─────────────────────────────────────────────────────────────────┐
│ [Enter symbol...                    ] [+ Add to Watchlist]      │
├──────┬──────┬────────┬────────────────┬─────────┬──────┬────────┤
│Symbol│Price │ Change │    52W Range   │ Volume  │Market│ Remove │
│      │      │        │                │         │ Cap  │        │
├──────┼──────┼────────┼────────────────┼─────────┼──────┼────────┤
│ AAPL │$268  │▲+0.48% │$150.12-$199.62 │48,227K  │$3.97T│   ×    │
│GOOGL │$278  │▼-2.08% │$130.45-$180.25 │34,479K  │$3.37T│   ×    │
└──────┴──────┴────────┴────────────────┴─────────┴──────┴────────┘
```

**Features:**
- Add symbols using input field
- Remove symbols with "×" button
- Real-time price updates
- Grid-based layout for perfect column alignment

**Columns:**
- Symbol (140px) - Stock ticker
- Price (130px) - Current price
- Change (130px) - Daily change ($ and %)
- 52W Range (200px) - 52-week high/low range
- Volume (150px) - Trading volume
- Market Cap (flexible) - Market capitalization
- Remove (button) - Delete from watchlist

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
- Lists all trained RL agents (PPO, RecurrentPPO, Ensemble)
- Checkbox selection for batch backtesting
- Shows agent type and symbol
- Training date
- Backtest button appears when models are selected
- Performance note: "Run backtest →" (metrics calculated on-demand)

**Batch Backtesting:**
- Select multiple models using checkboxes
- Backtest button shows count of selected models
- Run comprehensive backtests on multiple models and symbols simultaneously
- Compare performance across all selected models

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
- 4 algorithms: PPO, RecurrentPPO, DQN, QRDQN
- RecurrentPPO with LSTM memory and trend indicators
- Advanced risk management (stop-loss, trailing stops, circuit breakers)
- Market regime detection (BULL, BEAR, SIDEWAYS, VOLATILE)
- Multi-timeframe features (weekly/monthly trend analysis)
- Kelly position sizing (optimal sizing based on edge)
- Ensemble agents (combine multiple algorithms)
- Live training progress with real-time metrics
- Comprehensive backtesting engine
- Strategy comparison (All RL agents vs Buy&Hold vs Momentum)

### ✅ Live Trading Simulation
- Paper trading with real-time data
- Trained agent execution (PPO/RecurrentPPO/DQN/QRDQN)
- Real-time portfolio tracking
- Integrated risk management (stop-loss, trailing stops, circuit breakers)
- Market regime awareness
- Kelly position sizing
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
                            → Train agent (PPO/RecurrentPPO/DQN/QRDQN)
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

## UI File Structure

```
src/ui/
├── app.py                 # Main app factory (create_app, WatchlistPanel)
├── design_system.py       # Colors, HTML components, table styles
├── __init__.py            # Module exports
└── pages/
    ├── __init__.py        # Page module exports
    ├── analysis.py        # Analysis page (StockAnalysisApp)
    ├── dashboard.py       # Market overview page
    ├── rl_training.py     # RL training page (enhanced)
    ├── live_trading.py    # Live paper trading page (multi-session support)
    ├── portfolio.py       # Watchlist page (stock tracker)
    └── models.py          # Model registry page
```

**Key Files:**
- **app.py**: Application factory that creates the main layout, tabs, sidebar
- **design_system.py**: Shared design tokens (colors, styles, components)
- **pages/**: Individual page implementations for each tab

---

## Recent Improvements

### Live Trading Page
- **Multi-Session Support**: Dashboard now displays all sessions with aggregate metrics
- **Session Persistence**: Sessions auto-save every 60 seconds and resume on app restart
- **Wider Model Column**: Model name column increased from 200px to 300px to accommodate LSTM PPO naming
- **Smart Updates**: UI updates in-place without page refresh or scroll jumps
- **Environment Matching**: Sessions load exact training configuration from saved models

### Symbol Input
- **Unrestricted Input**: Symbol inputs now accept any valid ticker, not just predefined lists
- **Autocomplete Suggestions**: Predefined symbols shown as suggestions but don't restrict input
- **Applied To**: Analysis page, Training page, and Live Trading page

### Configuration System
- **Consistent Defaults**: All pages reference EnvConfig for default values (80% max position)
- **Single Source of Truth**: env_factory.py centralizes all environment parameters
- **No Duplication**: Changing defaults requires editing only one place

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
   - Uses polling (5-second intervals for live trading, 60-second trading cycles)
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
- Main app factory: `src/ui/app.py`
- Analysis page: `src/ui/pages/analysis.py`
- Page implementations: `src/ui/pages/`
