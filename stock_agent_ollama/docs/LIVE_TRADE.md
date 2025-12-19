# Live Trading Simulation Design

**Real-time RL Agent Trading with Live Market Data**

---

## Executive Summary

This document outlines the design for a **live trading simulation system** that uses trained RL agents to execute trades based on real-time market data. The system operates in a **simulated environment** with virtual capital, providing a safe and educational platform for testing algorithmic trading strategies.

**Key Objectives:**
- Execute RL agent decisions using real-time stock prices
- Track virtual portfolio performance in real-time
- Implement comprehensive risk management
- Provide live monitoring and alerts
- Maintain complete audit trail
- Educational and research-focused

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Trading Dashboard                    │
│  Real-time Portfolio • Performance Metrics • Trade Log       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Trading Execution Engine                    │
│  • Agent Inference      • Order Management                   │
│  • Position Tracking    • Risk Checks                        │
│  • Event Logging        • Performance Calc                   │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Market Data  │    │   RL Agent   │    │  Portfolio   │
│   Stream     │    │   (Trained)  │    │   Manager    │
│              │    │              │    │              │
│ • Yahoo API  │    │ • PPO/RPPO/  │    │ • Positions  │
│ • Real-time  │    │   Ensemble   │    │ • Cash       │
│ • 1-min bars │    │ • Inference  │    │ • P&L        │
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Risk Manager    │
                    │  • Stop Loss     │
                    │  • Position Lim  │
                    │  • Circuit Break │
                    └──────────────────┘
```

---

## Core Components

### 1. Market Data Stream

**Purpose:** Provide real-time price data to the trading engine

**Data Sources:**
- **Primary:** Yahoo Finance API (free, 1-minute delayed)
- **Fallback:** yfinance library real-time quotes
- **Frequency:** Poll every 60 seconds (configurable)

**Data Structure:**
```python
@dataclass
class MarketTick:
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
```

**Features:**
- Async data fetching
- Automatic reconnection on failure
- Data validation and sanitization
- Historical buffer for technical indicators
- Market hours detection

---

### 2. RL Agent Inference Engine

**Purpose:** Execute trained RL agent to generate trading decisions

**Process Flow:**
```
1. Receive market tick
2. Build observation state (price history, indicators, position)
3. Run agent.predict(observation)
4. Map action to trade decision
5. Send to order manager
```

**State Construction:**
```python
class ObservationBuilder:
    def build_observation(self, market_data: pd.DataFrame,
                         portfolio: Portfolio) -> np.ndarray:
        """
        Build observation vector for RL agent

        Components:
        - Price features (normalized returns, volatility)
        - Technical indicators (RSI, MACD, BB)
        - Position info (holdings, cash ratio)
        - LSTM features (if enabled for RecurrentPPO)
        """
```

**Action Mapping (6-Action Space):**
- **0:** HOLD (maintain current position)
- **1:** BUY_SMALL (invest ~15% of capital)
- **2:** BUY_MEDIUM (invest ~30% of capital)
- **3:** BUY_LARGE (invest ~50% of capital)
- **4:** SELL_PARTIAL (sell 50% of position)
- **5:** SELL_ALL (liquidate entire position)

---

### 3. Order Management System

**Purpose:** Execute trading decisions with validation and safety checks

**Order Flow:**
```
Agent Decision → Validation → Risk Check → Execution → Confirmation
```

**Order Types:**
```python
@dataclass
class Order:
    order_id: str
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    timestamp: datetime
    status: str  # 'PENDING', 'FILLED', 'REJECTED'
    reason: Optional[str]  # Rejection reason
```

**Validation Rules:**
1. **Sufficient Cash:** Verify available balance for buys
2. **Sufficient Shares:** Verify holdings for sells
3. **Price Sanity:** Check price within reasonable range
4. **Order Size:** Enforce min/max position sizes
5. **Market Hours:** Only trade during market hours (configurable)

**Execution Simulation:**
- Use current market price + slippage
- Apply transaction costs (0.1% default)
- Instant fills (no partial fills)
- Record all executions

---

### 4. Portfolio Manager

**Purpose:** Track positions, cash, and performance

**Portfolio State:**
```python
@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position]  # symbol -> Position
    initial_value: float
    total_value: float
    realized_pnl: float
    unrealized_pnl: float
    trades_count: int
    winning_trades: int
    losing_trades: int
```

**Position Tracking:**
```python
@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float
```

**Performance Metrics:**
- Total Return %
- Daily Return %
- Sharpe Ratio (rolling)
- Max Drawdown
- Win Rate
- Average Win/Loss
- Profit Factor

---

### 5. Risk Management System

**Purpose:** Protect capital with automated safety controls

**Integrated Risk Features:**

The trading environment includes advanced risk management components:

**A. Stop-Loss Manager** (`RiskManager` in `improvements.py`)
- **5% position stop-loss**: Exits if position down 5% from entry
- **3% trailing stop**: Locks in profits as position moves up
- **15% portfolio circuit breaker**: Halts trading if portfolio down 15%
- Automatically forces SELL_ALL when triggered

**B. Market Regime Detector** (`RegimeDetector` in `improvements.py`)
- Detects 4 market regimes: BULL, BEAR, SIDEWAYS, VOLATILE
- Adds 7 regime features to agent observations
- Helps agent adapt to market conditions

**C. Kelly Position Sizer** (`KellyPositionSizer` in `improvements.py`)
- Optimizes position sizes based on win rate and edge
- Uses Kelly Criterion for optimal sizing
- Requires minimum 20 trades before activating

### 6. Session Persistence

**Purpose:** To save and load the state of a live trading session, allowing it to be resumed across application restarts.

**Mechanism:**
- The state of the `TradingSession` object, including the `Portfolio`, `Position`, and `Trade` objects, is serialized to a JSON file.
- The session state is saved automatically:
    - Every 60 seconds while the session is running
    - When the "Stop Trading" button is clicked
    - Before application shutdown
- When the application starts, it checks for saved session files and loads them if found.

**State Files:**
- Location: `data/live_sessions/SESSION_*.json`
- Session ID formats:
  - Manual mode: `SESSION_SYMBOL_YYYYMMDD_HHMMSS.json` (e.g., `SESSION_AAPL_20251208_210831.json`)
  - Auto-select mode: `SESSION_AUTO_YYYYMMDD_HHMMSS.json` (e.g., `SESSION_AUTO_20251208_211108.json`)
- Format: JSON with complete session state including:
  - Configuration (symbol, algorithm, position limits, stop loss, auto_select_stock flag)
  - Portfolio (cash, positions, trades history)
  - Session metadata (start time, status, events including stock rotations)
- Multi-session support: Multiple sessions can be saved and resumed independently
- Session display: Sessions sorted by creation time (newest first) in UI

**Risk Controls:**

**A. Integrated Risk Management** (from `improvements.py`)

The environment includes built-in risk management via `RiskManager`, `RegimeDetector`, and `KellyPositionSizer`:

```python
# RiskManager (integrated into EnhancedTradingEnv)
use_risk_manager: bool = True
stop_loss_pct: float = 0.05        # 5% stop-loss
trailing_stop_pct: float = 0.03     # 3% trailing stop
max_drawdown_pct: float = 0.15      # 15% portfolio circuit breaker

# RegimeDetector (market condition awareness)
use_regime_detector: bool = True
# Detects: BULL, BEAR, SIDEWAYS, VOLATILE

# KellyPositionSizer (optimal position sizing)
use_kelly_sizing: bool = True
# Adjusts BUY sizes based on win rate and edge
```

**Additional Safety Controls:**

**Position Limits:**
```python
max_position_size: float = 0.80  # Max 80% of portfolio in one stock
max_total_exposure: float = 0.80  # Max 80% invested
min_cash_reserve: float = 0.20   # Keep 20% cash minimum
```

**Circuit Breakers:**
```python
# Daily Loss Limit
daily_loss_limit: float = -5.0%  # Stop trading if down 5% today

# Rapid Loss Protection
rapid_loss_threshold: float = -2.0%  # Trigger if loss > 2% in 5 min
rapid_loss_cooldown: int = 300  # Wait 5 min before resuming

# Volatility Filter
max_volatility: float = 0.05  # Skip trades if vol > 5%
```

**Time-Based Controls:**
```python
class TradingHours:
    market_open: time = time(9, 30)  # 9:30 AM
    market_close: time = time(16, 0)  # 4:00 PM

    # Trading days only
    trading_days: List[str] = ['MON', 'TUE', 'WED', 'THU', 'FRI']
```

---

## Auto Stock Selection

**Purpose:** Dynamically rotate trading between watchlist stocks to maximize capital efficiency

**How It Works:**

1. **Continuous Monitoring**: System evaluates rotation opportunities every cycle after cooldown expires
2. **Rotation Cooldown**: Waits minimum 10 cycles or 10 minutes before considering rotation (prevents premature switching)
3. **Idle Detection**: Tracks consecutive cycles where agent refuses to trade:
   - After 20 idle cycles, applies -50% penalty to current stock's performance score
   - Forces rotation to more active opportunities when agent signals disinterest
   - Prevents capital sitting idle while better trading opportunities exist
4. **Recency Penalty**: Prevents rapid "ping-pong" rotation between stocks:
   - Tracks when a stock was rotated away from
   - Applies -30% penalty to stocks visited within the last 30 minutes
   - Ensures rotation seeks fresh opportunities rather than cycling between two dormant stocks
5. **Shadow Mode Signal Scanning**: When portfolio is >50% cash and idle for 3+ cycles:
   - Actively checks top watchlist candidates for live BUY signals
   - Loads candidate models temporarily to test real-time trading intent
   - Rotates immediately to stocks showing active buying interest
   - Prioritizes actual agent signals over static historical metrics
6. **Stock Evaluation**: Analyzes all watchlist stocks including current symbol for performance:
   - Prioritizes agent backtest performance (total_return_pct from backtest_results.json)
   - Falls back to 5-day price performance if no backtest results available
7. **Model Selection**: Uses dynamic backtest-based scoring to find best model per stock:
   - Primary: Sharpe ratio × 20 + Return % × 2 (e.g., 2.5 Sharpe + 20% return = 90 points)
   - Fallback: Conservative algorithm preference (20-30 pts) + recency bonus (up to 10 pts) when no backtest
   - Training quality: Up to 20 points based on timesteps (300k+ = 20 pts, 100k+ = 10 pts)
8. **Position Closure**: Automatically closes current position before rotation to capture better opportunities
9. **Rotation Decision**: Only rotates if new stock is 2% better than current (prevents ping-pong effect)
10. **Single Session Policy**: Creating new AUTO session automatically stops existing AUTO sessions
11. **Agent Reload**: Automatically loads new stock's best model and reinitializes environment
12. **Event Logging**: Records force close and rotation with PnL in session event log

**Configuration:**
```python
config = LiveTradingConfig(
    auto_select_stock=True,  # Enable auto-rotation
    # Symbol and agent_path are dynamically updated
)
```

**Session Identification:**
- Auto-select sessions use ID format: `SESSION_AUTO_YYYYMMDD_HHMMSS`
- Manual sessions use format: `SESSION_SYMBOL_YYYYMMDD_HHMMSS`
- Auto-select sessions display cyan "AUTO" badge in session table

**Benefits:**
- Maximizes capital efficiency by always trading strongest performers
- No manual intervention required
- Automatically adapts to changing market conditions
- Leverages entire watchlist for opportunities
- Intelligent idle detection prevents capital stagnation
- Shadow mode actively seeks stocks with live BUY signals
- Recency penalty prevents ping-pong rotation between dormant stocks
- Intelligent cooldown prevents excessive rotation
- Performance threshold ensures meaningful switches
- Dynamic scoring selects best algorithm based on actual backtest performance
- Automatic position closure prevents missed opportunities
- Single session policy prevents capital fragmentation

**Requirements:**
- Trained models must exist for watchlist stocks
- Watchlist must contain at least 2 stocks
- Regular market hours only (9:30 AM - 4:00 PM ET)

**Event Logging:**
- All events prefixed with symbol name (e.g., `[HOOD] Agent predicted SELL_ALL...`)
- Rotation events show model name (e.g., `Rotated to HOOD (recurrent_ppo_HOOD_20251203_103214)`)
- Provides clear audit trail for multi-stock trading

---

## Live Trading Workflow

### Session Lifecycle

```
1. INITIALIZATION
   ├─ Check for saved session file
   ├─ If found, load session state (RESUME)
   ├─ If not found, start new session (NEW)
   ├─ Load trained RL agent
   ├─ Initialize portfolio (starting cash)
   ├─ Configure risk parameters
   ├─ Validate market hours
   └─ Start data stream

2. TRADING LOOP (every 60 seconds)
   ├─ Fetch market tick
   ├─ Update portfolio values
   ├─ Check risk controls
   ├─ Build observation
   ├─ Run agent inference
   ├─ Generate order (if action != HOLD)
   ├─ Validate order
   ├─ Execute order (if valid)
   ├─ Update portfolio
   ├─ Log trade
   └─ Update dashboard

3. SHUTDOWN
   ├─ Save session state
   ├─ Close all positions (optional)
   ├─ Generate performance report
   └─ Archive logs
```

### Decision Flow

```
Market Tick Received
        │
        ▼
Update Portfolio Values
        │
        ▼
Risk Check: Portfolio Stop Loss? ──YES──> HALT TRADING
        │ NO
        ▼
Risk Check: Daily Loss Limit? ──YES──> PAUSE (cooldown)
        │ NO
        ▼
Build Observation State
        │
        ▼
Agent Prediction (action, confidence)
        │
        ▼
Action == HOLD? ──YES──> Skip to next tick
        │ NO
        ▼
Generate Order
        │
        ▼
Validate Order
        │
        ├─ Sufficient funds? ──NO──> REJECT
        ├─ Position limits? ──NO──> REJECT
        ├─ Price valid? ──NO──> REJECT
        └─ Market hours? ──NO──> REJECT
        │ ALL YES
        ▼
Execute Order (simulate fill)
        │
        ▼
Update Portfolio
        │
        ▼
Log Trade + Update Metrics
        │
        ▼
Update Dashboard
```

---

## Data Models

### 1. Trading Session

```python
@dataclass
class TradingSession:
    session_id: str
    agent_path: str
    symbol: str
    start_time: datetime
    end_time: Optional[datetime]
    initial_capital: float

    # Configuration
    config: LiveTradingConfig

    # State
    status: str  # 'ACTIVE', 'PAUSED', 'STOPPED', 'HALTED'
    portfolio: Portfolio

    # Performance
    metrics: PerformanceMetrics

    # History
    trades: List[Trade]
    market_ticks: List[MarketTick]
    events: List[TradingEvent]
```

### 2. Live Trading Config

```python
@dataclass
class LiveTradingConfig:
    # Agent
    symbol: str
    agent_path: str

    # Portfolio
    initial_capital: float = 100000.0
    transaction_cost: float = 0.0005  # 0.05%
    slippage: float = 0.0005  # 0.05%

    # Execution
    poll_interval: int = 60  # seconds
    enforce_trading_hours: bool = True
    auto_select_stock: bool = False  # Dynamically rotate between best stocks

    # Risk Management
    stop_loss_pct: float = 5.0  # 5% stop loss
    max_position_size: float = 80.0  # Max 80% of portfolio
    max_portfolio_risk_pct: float = 2.0  # Max 2% portfolio risk per trade
    max_daily_loss_pct: float = 10.0  # 10% max daily loss (circuit breaker)

    # Session
    session_id: Optional[str] = None
    commission_per_trade: float = 0.0
```

### 3. Trade Record

```python
@dataclass
class Trade:
    trade_id: str
    timestamp: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    cost: float  # Including fees
    agent_action: int  # Raw agent output
    agent_confidence: float

    # Context
    portfolio_value_before: float
    portfolio_value_after: float
    position_size_pct: float

    # Performance (for closed positions)
    realized_pnl: Optional[float]
    holding_period: Optional[int]  # minutes
```

### 4. Trading Event

```python
@dataclass
class TradingEvent:
    timestamp: datetime
    event_type: str  # 'TRADE', 'STOP_LOSS', 'CIRCUIT_BREAKER', etc.
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    message: str
    data: Dict[str, Any]
```

---

## User Interface

### Live Trading Dashboard

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Live Trading Simulation - AAPL                 [PAUSE] [STOP]│
│  Status: ACTIVE    Runtime: 2h 34m    Last Update: 10:45:23  │
├─────────────────────────────────────────────────────────────┤
│  Portfolio Summary                                           │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │ Total Value  │ Cash         │ Invested     │ Today P&L │ │
│  │ $10,523.45   │ $4,200.00    │ $6,323.45    │ +$523.45  │ │
│  │ +5.23%       │ 39.9%        │ 60.1%        │ +5.23%    │ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Current Position                                            │
│  AAPL: 25 shares @ $252.94 avg                              │
│  Current: $254.12 (+$29.50, +1.18%)                         │
│  Stop Loss: $240.29 (-5.0%)                                 │
├─────────────────────────────────────────────────────────────┤
│  Performance Metrics                                         │
│  Total Return: +5.23%  |  Sharpe: 1.45  |  Max DD: -2.1%   │
│  Trades: 12  |  Win Rate: 66.7%  |  Avg Win: +1.2%          │
├─────────────────────────────────────────────────────────────┤
│  [Portfolio Value Chart - Real-time Line Graph]             │
├─────────────────────────────────────────────────────────────┤
│  Recent Trades (Symbol shown for auto-select sessions)      │
│  2025-12-12 10:43:12  AAPL  BUY   5 @ $253.45  P&L: $0.00  │
│  2025-12-12 10:15:34  AAPL  SELL  3 @ $251.20  P&L: $6.75  │
│  2025-12-12 09:52:18  NVDA  BUY  10 @ $249.80  P&L: $0.00  │
├─────────────────────────────────────────────────────────────┤
│  Risk Status: ✅ All Clear                                   │
│  • Position Size: 60.1% (max 80%)                           │
│  • Daily Loss: +5.23% (limit -5%)                           │
│  • Stop Loss: $240.29 (current $254.12)                     │
└─────────────────────────────────────────────────────────────┘
```

**Interactive Controls:**
- **Pause/Resume:** Stop execution without closing positions
- **Stop:** Close all positions and end session
- **Emergency Stop:** Immediate halt with position liquidation
- **Adjust Risk:** Modify stop-loss levels on the fly
- **Export Data:** Download session data as CSV/JSON

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

**Components:**
1. Market data stream with real-time polling
2. Basic portfolio tracking
3. Simple order execution (buy/sell at market)
4. Trade logging to database/CSV

**Deliverables:**
- `LiveTradingEngine` class
- `MarketDataStream` class
- `Portfolio` class
- `OrderExecutor` class

**Testing:**
- Unit tests for each component
- Integration test with paper trading
- Validate order execution logic

---

### Phase 2: Risk Management (Week 3)

**Components:**
1. Stop-loss implementation (position + portfolio)
2. Position size limits
3. Circuit breakers
4. Risk event logging

**Deliverables:**
- `RiskManager` class
- `CircuitBreaker` class
- Risk monitoring dashboard panel

**Testing:**
- Trigger stop-loss scenarios
- Test circuit breaker activation
- Verify position limits enforcement

---

### Phase 3: Agent Integration (Week 4)

**Components:**
1. Load trained RL agents
2. Build observation from live data
3. Execute agent predictions
4. Handle LSTM features (if enabled)

**Deliverables:**
- `AgentInference` class
- `ObservationBuilder` class
- Agent-to-order translation

**Testing:**
- Test with all agent types (PPO, RecurrentPPO, Ensemble)
- Validate observation construction
- Compare backtest vs live predictions

---

### Phase 4: Dashboard & Monitoring (Week 5)

**Components:**
1. Real-time portfolio display
2. Live performance charts
3. Trade log with filters
4. Alert system

**Deliverables:**
- Live trading UI page in Panel
- Real-time chart updates
- Alert notifications
- Session management controls

**Testing:**
- UI responsiveness with live updates
- Chart rendering performance
- Alert delivery verification

---

### Phase 5: Analytics & Reporting (Week 6)

**Components:**
1. Session replay functionality
2. Performance analysis tools
3. Trade-by-trade breakdown
4. Export capabilities

**Deliverables:**
- Session analytics module
- Performance comparison tools
- Export to CSV/JSON/Excel

**Testing:**
- Validate metric calculations
- Test export formats
- Performance regression tests

---

## Key Design Decisions & Bug Fixes

### Configuration Management
**Single Source of Truth:** All environment parameters (transaction costs, position limits, slippage, etc.) are centralized in `env_factory.py` with the `EnvConfig` dataclass. Training, backtesting, and live trading all reference this single source, preventing train-test mismatch.

**Environment Loading:** Live trading automatically loads the exact environment configuration used during training from saved `training_config.json` files, ensuring consistent behavior between training and deployment.

### Critical Bug Fixes
**Short-Selling Prevention:** Fixed bug in `AdaptiveActionSizer.get_sell_size()` where selling with zero position would return 1 share, creating unintentional short positions. Now properly checks `if position == 0: return 0`.

**Floating Point Tolerance:** Added 0.01% tolerance to position size checks to prevent legitimate orders at exactly the limit (e.g., 80.0%) from being rejected due to floating point precision issues.

**Action Mapping:** Corrected action enums to use `TradingAction` throughout instead of non-existent `OrderAction`, ensuring proper action execution.

## Technical Implementation

### Core Classes

#### 1. LiveTradingEngine

**Implementation**: `src/rl/live_trading.py`

```python
class LiveTradingEngine:
    """Main orchestrator for live trading simulation"""

    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.session = TradingSession(...)
        self.portfolio = Portfolio(initial_cash=config.initial_capital)
        self.market_stream = MarketDataStream(config.symbol)
        self.agent = self._load_agent(config.agent_path)
        self.risk_manager = RiskManager(config)
        self.order_executor = OrderExecutor(config)
        self.running = False

    def start(self):
        """Start live trading session"""
        self.running = True
        self.session.start_time = datetime.now()
        self.session.status = 'ACTIVE'

        # Main trading loop
        while self.running:
            try:
                # Wait for next poll interval
                time.sleep(self.config.poll_interval)

                # Execute trading cycle
                self._trading_cycle()

            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                self._handle_error(e)

    def _trading_cycle(self):
        """Single iteration of trading logic"""
        # 1. Fetch market data
        tick = self.market_stream.get_latest_tick()

        # 2. Update portfolio valuations
        self.portfolio.update_valuations(tick)

        # 3. Check risk controls
        risk_status = self.risk_manager.check(self.portfolio, tick)
        if risk_status.halt:
            self._halt_trading(risk_status.reason)
            return
        if risk_status.pause:
            logger.warning(f"Trading paused: {risk_status.reason}")
            return

        # 4. Build observation for agent
        observation = self._build_observation(tick)

        # 5. Get agent prediction
        action, confidence = self.agent.predict(observation)

        # 6. Translate action to order
        if action == TradingAction.HOLD:
            return

        order = self._create_order(action, tick, confidence)

        # 7. Validate and execute
        if self.order_executor.validate(order, self.portfolio):
            trade = self.order_executor.execute(order, self.portfolio)
            self.session.trades.append(trade)
            logger.info(f"Trade executed: {trade}")
        else:
            logger.warning(f"Order rejected: {order}")

    def stop(self, close_positions=True):
        """Stop trading session"""
        self.running = False
        self.session.status = 'STOPPED'
        self.session.end_time = datetime.now()

        if close_positions:
            self._close_all_positions()

        self._save_session()
        self._generate_report()
```

#### 2. MarketDataStream

**Implementation**: `src/rl/live_trading.py`

```python
class MarketDataStream:
    """Real-time market data provider"""

    def __init__(self, symbol: str, history_size: int = 100):
        self.symbol = symbol
        self.fetcher = StockFetcher()
        self.tick_history = deque(maxlen=history_size)

    def get_latest_tick(self) -> MarketTick:
        """Fetch latest market data"""
        data = self.fetcher.get_real_time_price(self.symbol)

        tick = MarketTick(
            symbol=self.symbol,
            timestamp=datetime.now(),
            price=data['current_price'],
            volume=data.get('volume', 0),
            bid=None,  # Not available in free API
            ask=None,
            spread=None
        )

        self.tick_history.append(tick)
        return tick

    def get_history_df(self, lookback: int = 50) -> pd.DataFrame:
        """Get recent tick history as DataFrame"""
        recent_ticks = list(self.tick_history)[-lookback:]
        return pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'price': t.price,
                'volume': t.volume
            }
            for t in recent_ticks
        ])
```

#### 3. RiskManager

**Implementation**: `src/rl/live_trading.py`

```python
class RiskManager:
    """Automated risk control system"""

    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.stop_loss_triggered = False
        self.circuit_breaker_active = False
        self.circuit_breaker_cooldown_until = None

    def check(self, portfolio: Portfolio, tick: MarketTick) -> RiskStatus:
        """Run all risk checks"""
        status = RiskStatus(halt=False, pause=False, reason=None)

        # Portfolio-wide stop loss
        if portfolio.total_return_pct <= self.config.portfolio_stop_loss:
            status.halt = True
            status.reason = f"Portfolio stop loss triggered: {portfolio.total_return_pct:.2%}"
            return status

        # Daily loss limit
        if portfolio.daily_return_pct <= self.config.daily_loss_limit:
            status.pause = True
            status.reason = f"Daily loss limit: {portfolio.daily_return_pct:.2%}"
            return status

        # Circuit breaker cooldown
        if self.circuit_breaker_cooldown_until:
            if datetime.now() < self.circuit_breaker_cooldown_until:
                status.pause = True
                status.reason = "Circuit breaker cooldown active"
                return status
            else:
                self.circuit_breaker_cooldown_until = None

        # Check position-specific stop losses
        for symbol, position in portfolio.positions.items():
            if position.unrealized_pnl_pct <= self.config.position_stop_loss:
                logger.warning(f"Position stop loss for {symbol}: {position.unrealized_pnl_pct:.2%}")
                # Trigger sell order (handled by engine)

        return status
```

---

## Safety & Best Practices

### Educational Disclaimers

**All UI Elements Must Display:**
```
⚠️ SIMULATION ONLY - NOT REAL TRADING
This is a paper trading system using virtual money for
educational purposes. No real trades are executed.
```

### Data Validation

1. **Price Sanity Checks:**
   - Reject prices outside ±10% of previous tick
   - Detect and handle market halts
   - Validate data freshness (< 5 min old)

2. **State Validation:**
   - Portfolio value = cash + positions (must balance)
   - No negative cash (after accounting for orders)
   - Position quantities match trade history

3. **Agent Validation:**
   - Verify model compatibility with current data
   - Check for model staleness (training date)
   - Validate observation dimensions

### Logging & Audit Trail

**Required Logs:**
- Every trade (timestamp, action, price, quantity)
- All risk events (stop-loss, circuit breakers)
- Agent decisions (action, confidence, observation)
- System errors and exceptions
- Session start/stop events

**Log Format:**
```json
{
  "timestamp": "2025-01-03T10:45:23.123Z",
  "event_type": "TRADE",
  "session_id": "session_20250103_104512",
  "symbol": "AAPL",
  "action": "BUY",
  "quantity": 10,
  "price": 253.45,
  "cost": 2534.75,
  "portfolio_value": 10523.45,
  "agent_action": 3,
  "agent_confidence": 0.87
}
```

---

## Performance Considerations

### Optimization Strategies

1. **Async Data Fetching:**
   - Non-blocking market data requests
   - Concurrent multi-symbol support
   - Background indicator calculation

2. **Caching:**
   - Cache recent market data
   - Reuse observation calculations
   - Store computed indicators

3. **Database:**
   - Use SQLite for trade logs
   - Batch inserts for performance
   - Indexed queries for analysis

4. **UI Updates:**
   - Optimized granular DOM updates via row caching
   - Efficient state hashing for minimal change detection
   - Zero-flicker refresh for high session counts
   - Lazy loading for detailed history views

---

## Testing Strategy

### Unit Tests

- Portfolio calculations
- Order validation logic
- Risk control triggers
- Agent inference
- Data transformations

### Integration Tests

- Full trading cycle execution
- Market data stream reliability
- Risk manager integration
- Database operations

### Simulation Tests

- 1-day fast-forward simulation
- Multi-symbol trading
- Extreme market conditions
- Recovery from errors

### Stress Tests

- High-frequency ticks (1-second interval)
- Large position sizes
- Multiple simultaneous trades
- Network failures and retries

---

## Deployment

### Configuration

**Via Web UI:**
All configuration is done through the Live Trade page interface:
- Symbol selection
- Algorithm selection (PPO/RecurrentPPO/Ensemble)
- Initial capital
- Max position size
- Stop loss percentage

**Persistence:**
Configuration is saved with the session state in:
- `data/live_sessions/live_session.json`

**Default Settings:**
- Initial capital: $100,000
- Transaction cost: 0.05%
- Stop loss: 5%
- Poll interval: 60 seconds

### Running a Session

**Via Web UI (Recommended):**
1. Launch the application: `python src/main.py`
2. Navigate to **Live Trade** tab
3. Configure settings (PPO/RecurrentPPO/Ensemble) and click "Start Trading"
4. Sessions are automatically saved and can be resumed on restart

**Session State:**
- Saved to: `data/live_sessions/live_session.json`
- Auto-saved every 5 minutes while active
- Automatically loaded on application start
- Contains portfolio, positions, trades, and configuration

---

## Future Enhancements

### Short-term
- [ ] Multi-symbol portfolio support
- [ ] Email/SMS alerts for key events
- [ ] Session replay with visualization
- [ ] Advanced order types (limit, stop-limit)

### Medium-term
- [ ] Machine learning-based risk adjustment
- [ ] Reinforcement learning from live experience
- [ ] Integration with broker APIs (paper trading)
- [ ] Mobile app for monitoring

### Long-term
- [ ] Multi-agent ensemble trading
- [ ] Market regime detection
- [ ] Automated strategy optimization
- [ ] Social trading features (share strategies)

---

## Conclusion

The **Live Trading Simulation** system provides a safe, educational platform for testing RL trading strategies with real-time market data. By combining trained agents, comprehensive risk management, and live monitoring, users can evaluate algorithmic trading performance in a realistic environment without financial risk.

**Key Advantages:**
- ✅ Real-time market data integration
- ✅ Automated risk controls
- ✅ Complete audit trail
- ✅ Educational and research-focused
- ✅ No real financial risk

**Educational Purpose:**
This system is designed for learning, research, and algorithm validation. It is **not intended for actual trading** and should not be used to make real investment decisions.