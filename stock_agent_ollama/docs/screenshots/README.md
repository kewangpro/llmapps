# Screenshots Directory

This directory contains screenshots for the README documentation.

## Required Screenshots

### `analysis.png`
**Description**: Stock Analysis Tab screenshot showing:
- Stock chart with interactive candlestick patterns
- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- LSTM 30-day prediction with confidence intervals
- Trading signals (BUY/SELL/HOLD) with confidence scores
- AI-powered analysis with natural language insights

**Recommended size**: 1200x800px or higher
**Format**: PNG with good compression

**How to capture**:
1. Launch the application: `python src/main.py`
2. Navigate to Analysis tab
3. Select a stock symbol (e.g., AAPL, GOOGL, NVDA)
4. Click "Analyze" button
5. Wait for LSTM predictions and charts to load
6. Take screenshot of the full tab view showing charts, signals, and AI analysis
7. Save as `analysis.png`

---

### `training.png`
**Description**: RL Trading Tab screenshot showing:
- Training configuration interface OR
- Backtesting results with:
  - Performance metrics table (RL Agents vs Buy & Hold vs Momentum)
  - Action distribution visualization
  - Portfolio value/equity curves comparison
  - Stock price with trade signals overlay
  - Key metrics comparison chart

**Recommended size**: 1200x800px or higher
**Format**: PNG with good compression

**How to capture**:
1. Launch the application: `python src/main.py`
2. Navigate to Trading tab
3. **Option A (Training view)**:
   - Select stock symbol (e.g., AAPL, NVDA)
   - Choose algorithm (PPO, RecurrentPPO, or Ensemble)
   - Set training parameters
   - Capture the training interface with parameters visible
4. **Option B (Backtesting results view)** - RECOMMENDED:
   - Run a backtest with "Run Backtest" button
   - Wait for results to load
   - Capture the complete results view with charts, metrics table, and action distributions
5. Save as `training.png`

---

### `live_trade.png`
**Description**: Live Trading Tab screenshot showing:
- Multi-session dashboard with aggregate metrics
- Session creation panel
- Active session monitoring (status, portfolio, positions)
- Recent trades table with timestamps, symbols, costs, and P&L
- Event log showing trading activity
- Real-time portfolio value updates

**Recommended size**: 1200x800px or higher
**Format**: PNG with good compression

**How to capture**:
1. Launch the application: `python src/main.py`
2. Navigate to Live Trade tab
3. Create and start a live trading session (or resume existing session)
4. Wait for a few trading cycles to generate some activity
5. Capture the full view showing:
   - Dashboard with sessions table
   - Active session details
   - Portfolio summary
   - Positions table
   - Recent trades with costs
   - Event log
6. Save as `live_trade.png`

---

### `models.png`
**Description**: Models Page screenshot showing:
- Tabbed interface with dynamic header
- LSTM Models tab with trained prediction models
- RL Agents tab with PPO/RecurrentPPO/Ensemble models
- Model metadata (training dates, algorithms, symbols)
- Checkbox selection for batch backtesting
- Chronological ordering (newest first)

**Recommended size**: 1200x800px or higher
**Format**: PNG with good compression

**How to capture**:
1. Launch the application: `python src/main.py`
2. Navigate to Models tab
3. Switch between LSTM Models and RL Agents tabs to show both
4. Capture the tab showing RL Agents with some models listed and checkboxes
5. Save as `models.png`

---

### `dashboard.png` (Optional)
**Description**: Dashboard Page screenshot showing:
- Market overview with major indices (S&P 500, NASDAQ, Dow Jones, Russell 2000)
- Interactive watchlist in sidebar with live prices
- Quick actions panel
- Clean light theme interface

**Recommended size**: 1200x800px or higher
**Format**: PNG with good compression

**How to capture**:
1. Launch the application: `python src/main.py`
2. Navigate to Dashboard tab (should open by default)
3. Wait for market data to load
4. Capture the full view including sidebar watchlist
5. Save as `dashboard.png`

---

## Notes
- Screenshots should show realistic data and typical usage scenarios
- Ensure UI elements are clearly visible and readable
- Use high-resolution displays for better quality
- Crop to show relevant content without excessive whitespace
- Include enough context to understand the feature being demonstrated
- Light theme should be visible in all screenshots
- Show actual data rather than placeholder text when possible
