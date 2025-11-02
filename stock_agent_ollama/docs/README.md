# Screenshots Directory

This directory contains screenshots for the README documentation.

## Required Screenshots

### `analysis.png`
**Description**: Stock Analysis Tab screenshot showing:
- Natural language query interface
- AI-generated analysis response
- LSTM 30-day prediction chart
- Technical indicators (RSI, MACD, Bollinger Bands)
- Trading signals (BUY/SELL/HOLD)
- Interactive Plotly charts

**Recommended size**: 1200x800px or higher
**Format**: PNG with good compression

**How to capture**:
1. Launch the application: `python src/main.py`
2. Navigate to Stock Analysis tab
3. Enter query: "Analyze Apple stock performance"
4. Wait for LSTM predictions and charts to load
5. Take screenshot of the full tab view
6. Save as `analysis.png`

---

### `rl_trading.png`
**Description**: RL Trading Tab screenshot showing:
- Training interface with parameters (symbol, algorithm, timesteps)
- OR Backtesting results with:
  - Performance metrics table (RL Agent vs Baselines)
  - Action distribution bar chart
  - Portfolio value/equity curves
  - Drawdown charts
  - Action counts table

**Recommended size**: 1200x800px or higher
**Format**: PNG with good compression

**How to capture**:
1. Launch the application: `python src/main.py`
2. Navigate to RL Trading tab
3. **Option A (Training view)**:
   - Select stock symbol (e.g., AAPL)
   - Choose PPO algorithm
   - Set training parameters
   - Capture the training interface
4. **Option B (Backtesting results view)** - RECOMMENDED:
   - Run a backtest with "Run Backtest" button
   - Wait for results to load
   - Capture the complete results view with charts and metrics
5. Save as `rl_trading.png`

---

## Notes
- Screenshots should show realistic data and typical usage scenarios
- Ensure UI elements are clearly visible and readable
- Use high-resolution displays for better quality
- Crop to show relevant content without excessive whitespace
- Include enough context to understand the feature
