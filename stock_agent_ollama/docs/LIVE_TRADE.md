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
│ • Yahoo API  │    │ • PPO/A2C    │    │ • Positions  │
│ • Real-time  │    │ • LSTM feat  │    │ • Cash       │
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
        - LSTM features (if enabled)
        """
```

**Action Mapping:**
- **0:** SELL (liquidate position)
- **1:** HOLD (maintain current position)
- **2:** BUY_SMALL (invest 10% of capital)
- **3:** BUY_LARGE (invest 30% of capital)

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

**Risk Controls:**

**A. Stop-Loss Mechanisms:**
```python
class StopLossManager:
    # Per-Position Stop Loss
    position_stop_loss: float = -5.0%  # Exit if position down 5%

    # Portfolio-Wide Stop Loss
    portfolio_stop_loss: float = -10.0%  # Halt trading if down 10%

    # Trailing Stop Loss
    trailing_stop: float = 3.0%  # Lock in profits
```

**B. Position Limits:**
```python
class PositionLimits:
    max_position_size: float = 0.40  # Max 40% of portfolio in one stock
    max_total_exposure: float = 0.80  # Max 80% invested
    min_cash_reserve: float = 0.20  # Keep 20% cash minimum
```

**C. Circuit Breakers:**
```python
class CircuitBreakers:
    # Daily Loss Limit
    daily_loss_limit: float = -5.0%  # Stop trading if down 5% today

    # Rapid Loss Protection
    rapid_loss_threshold: float = -2.0%  # Trigger if loss > 2% in 5 min
    rapid_loss_cooldown: int = 300  # Wait 5 min before resuming

    # Volatility Filter
    max_volatility: float = 0.05  # Skip trades if vol > 5%
```

**D. Time-Based Controls:**
```python
class TradingHours:
    market_open: time = time(9, 30)  # 9:30 AM
    market_close: time = time(16, 0)  # 4:00 PM

    # Avoid first/last minutes
    avoid_first_minutes: int = 15
    avoid_last_minutes: int = 15

    # Trading days only
    trading_days: List[str] = ['MON', 'TUE', 'WED', 'THU', 'FRI']
```

---

## Live Trading Workflow

### Session Lifecycle

```
1. INITIALIZATION
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
   ├─ Close all positions (optional)
   ├─ Save session data
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
    agent_path: str
    agent_type: str  # 'ppo' or 'a2c'
    use_lstm: bool

    # Portfolio
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Execution
    poll_interval: int = 60  # seconds
    market_hours_only: bool = True

    # Risk Management
    position_stop_loss: float = -0.05
    portfolio_stop_loss: float = -0.10
    max_position_size: float = 0.40
    max_exposure: float = 0.80
    min_cash: float = 0.20

    # Circuit Breakers
    daily_loss_limit: float = -0.05
    rapid_loss_threshold: float = -0.02
    cooldown_period: int = 300
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
│  Recent Trades                                               │
│  10:43:12  BUY   5 @ $253.45  [Agent: BUY_SMALL, conf=0.82] │
│  10:15:34  SELL  3 @ $251.20  [Agent: SELL, conf=0.91]      │
│  09:52:18  BUY  10 @ $249.80  [Agent: BUY_LARGE, conf=0.88] │
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
- Test with multiple agent types (PPO, A2C)
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

## Technical Implementation

### Core Classes

#### 1. LiveTradingEngine

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
   - Throttle dashboard updates (max 1/sec)
   - Lazy load trade history
   - Virtualized scrolling for large lists

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

### Configuration Files

**`config/live_trading.yaml`:**
```yaml
session:
  name: "AAPL Live Trading"
  symbol: "AAPL"
  initial_capital: 10000.0

agent:
  path: "data/models/rl/ppo_AAPL_20250103_104523/final_model.zip"
  type: "ppo"
  use_lstm: true

execution:
  poll_interval: 60  # seconds
  market_hours_only: true
  transaction_cost: 0.001
  slippage: 0.0005

risk:
  position_stop_loss: -0.05
  portfolio_stop_loss: -0.10
  max_position_size: 0.40
  max_exposure: 0.80
  daily_loss_limit: -0.05

monitoring:
  enable_alerts: true
  alert_email: null
  log_level: "INFO"
```

### Running a Session

```bash
# Start live trading simulation
python src/live_trade.py --config config/live_trading.yaml

# Resume a paused session
python src/live_trade.py --resume session_20250103_104512

# Analyze past session
python src/live_trade.py --analyze session_20250103_104512
```

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

---

**Version:** 1.0
**Last Updated:** 2025-01-03
**Status:** Design Phase
**Next Steps:** Begin Phase 1 implementation (Core Infrastructure)
