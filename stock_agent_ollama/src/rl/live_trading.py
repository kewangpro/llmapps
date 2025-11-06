"""
Live Trading Simulation System

Educational paper trading system that uses trained RL models with real-time market data.
This is for SIMULATION ONLY - no real money is involved.
"""

from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import logging
import json



logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class TradingAction(Enum):
    """Trading actions"""
    BUY = 1
    SELL = 2
    HOLD = 0


class TradingStatus(Enum):
    """Trading session status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    HALTED = "halted"  # Emergency stop


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class MarketTick:
    """Single market data tick"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class Position:
    """Trading position"""
    symbol: str
    shares: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, new_price: float):
        """Update position with new market price"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.avg_entry_price) * self.shares

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(**data)


@dataclass
class Order:
    """Trading order"""
    symbol: str
    action: TradingAction
    shares: int
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['action'] = self.action.value
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        data['action'] = TradingAction(data['action'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['status'] = OrderStatus(data['status'])
        return cls(**data)


@dataclass
class Trade:
    """Executed trade"""
    symbol: str
    action: TradingAction
    shares: int
    price: float
    timestamp: datetime
    pnl: float = 0.0
    commission: float = 0.0
    trade_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['action'] = self.action.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        data['action'] = TradingAction(data['action'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Portfolio:
    """Trading portfolio"""
    initial_cash: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        """Total portfolio value"""
        position_value = sum(p.shares * p.current_price for p in self.positions.values())
        return self.cash + position_value

    @property
    def total_pnl(self) -> float:
        """Total profit/loss"""
        return self.total_value - self.initial_cash

    @property
    def total_pnl_pct(self) -> float:
        """Total P&L percentage"""
        return (self.total_pnl / self.initial_cash) * 100 if self.initial_cash > 0 else 0.0

    def update_valuations(self, tick: MarketTick):
        """Update position valuations with latest market data"""
        if tick.symbol in self.positions:
            self.positions[tick.symbol].update_price(tick.price)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "trades": [t.to_dict() for t in self.trades],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Portfolio":
        positions = {k: Position.from_dict(v) for k, v in data.get("positions", {}).items()}
        trades = [Trade.from_dict(t) for t in data.get("trades", [])]
        return cls(
            initial_cash=data["initial_cash"],
            cash=data["cash"],
            positions=positions,
            trades=trades,
        )


@dataclass
class LiveTradingConfig:
    """Live trading configuration"""
    symbol: str
    agent_path: str
    initial_capital: float = 10000.0
    max_position_size: int = 100
    max_portfolio_risk_pct: float = 2.0  # Max 2% portfolio risk per trade
    stop_loss_pct: float = 5.0  # 5% stop loss
    max_daily_loss_pct: float = 10.0  # 10% max daily loss (circuit breaker)
    update_interval: int = 60  # Seconds between updates
    enforce_trading_hours: bool = True
    allow_extended_hours: bool = False  # Default to False as requested
    commission_per_trade: float = 0.0  # Commission per trade
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiveTradingConfig":
        # Get all field names from the dataclass
        known_fields = {f.name for f in fields(cls)}
        # Filter data to only include known fields
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


@dataclass
class TradingSession:
    """Live trading session"""
    session_id: str
    config: LiveTradingConfig
    portfolio: Portfolio
    status: TradingStatus = TradingStatus.IDLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    events: List[Dict] = field(default_factory=list)

    def add_event(self, event_type: str, message: str):
        """Add event to session log"""
        self.events.append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "events": self.events, # events are already dicts
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSession":
        return cls(
            session_id=data["session_id"],
            config=LiveTradingConfig.from_dict(data["config"]),
            portfolio=Portfolio.from_dict(data["portfolio"]),
            status=TradingStatus(data["status"]),
            start_time=datetime.fromisoformat(data["start_time"]) if data["start_time"] else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            events=data.get("events", []),
        )


# ============================================================================
# Market Data Stream
# ============================================================================

class MarketDataStream:
    """Real-time market data stream using Yahoo Finance"""

    def __init__(self, config: LiveTradingConfig):
        self.symbol = config.symbol
        self.config = config
        self._last_tick: Optional[MarketTick] = None

    def get_latest_tick(self) -> MarketTick:
        """Get latest market tick with real-time updates"""
        try:
            # Create a fresh ticker instance to bypass yfinance caching
            ticker = yf.Ticker(self.symbol)

            # Use history with 1-minute interval for most recent data
            # This is more reliable than .info which caches heavily
            hist = ticker.history(period='1d', interval='1m')

            if hist.empty:
                # Fallback to daily data
                hist = ticker.history(period='1d', interval='1d')

            if hist.empty:
                raise ValueError(f"Unable to get price data for {self.symbol}")

            # Get the most recent row
            latest = hist.iloc[-1]
            current_price = float(latest['Close'])
            volume = int(latest['Volume']) if 'Volume' in latest else 0
            timestamp = latest.name  # Get the timestamp from the index

            # Try to get bid/ask from fast_info
            bid = None
            ask = None
            try:
                fast_info = ticker.fast_info
                if hasattr(fast_info, 'last_price'):
                    # Update price if fast_info has newer data
                    if fast_info.last_price and fast_info.last_price > 0:
                        current_price = float(fast_info.last_price)
            except:
                pass

            tick = MarketTick(
                symbol=self.symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=volume,
                bid=bid,
                ask=ask
            )

            self._last_tick = tick
            logger.debug(f"Fetched tick: {self.symbol} @ ${current_price:.2f} (data from {timestamp})")
            return tick

        except Exception as e:
            logger.error(f"Error fetching market data for {self.symbol}: {e}")
            # Return last known tick if available
            if self._last_tick:
                logger.warning(f"Using last known tick for {self.symbol}")
                return self._last_tick
            raise

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            # Create a fresh ticker to get the most up-to-date info
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            market_state = info.get('marketState', 'CLOSED').upper()
            
            if self.config.allow_extended_hours:
                # Valid states for extended hours trading
                return market_state in ['REGULAR', 'PRE', 'POST', 'PREPRE', 'POSTPOST']
            else:
                # Regular hours only
                return market_state == 'REGULAR'

        except Exception as e:
            logger.warning(f"Could not fetch market state from yfinance: {e}. Falling back to time-based check.")
            # Default to checking trading hours (e.g., 9:30 AM - 4:00 PM ET)
            try:
                from zoneinfo import ZoneInfo
                from datetime import datetime, time

                et_zone = ZoneInfo('US/Eastern')
                now_et = datetime.now(et_zone)

                if now_et.weekday() >= 5:  # Weekend
                    return False

                # Regular market hours: 9:30 AM to 4:00 PM ET
                market_open = time(9, 30) <= now_et.time() < time(16, 0)

                if not self.config.allow_extended_hours:
                    return market_open

                # Extended hours (pre-market and post-market)
                # Pre-market: 4:00 AM to 9:30 AM ET
                pre_market = time(4, 0) <= now_et.time() < time(9, 30)
                # Post-market: 4:00 PM to 8:00 PM ET
                post_market = time(16, 0) <= now_et.time() < time(20, 0)

                return market_open or pre_market or post_market

            except ImportError:
                logger.warning("zoneinfo not available (requires Python 3.9+). Falling back to simplified time check.")
                # Fallback for older Python
                now = datetime.now()
                if now.weekday() >= 5:  # Weekend
                    return False
                hour = now.hour
                # This is a rough approximation and assumes server time is close to ET.
                return 9 <= hour < 16


# ============================================================================
# Risk Manager
# ============================================================================

@dataclass
class RiskStatus:
    """Risk check result"""
    approved: bool
    halt: bool = False
    reason: Optional[str] = None


class RiskManager:
    """Risk management system"""

    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.daily_start_value: Optional[float] = None
        self.daily_low_value: Optional[float] = None

    def check(self, portfolio: Portfolio, tick: MarketTick) -> RiskStatus:
        """Comprehensive risk check"""

        # Initialize daily tracking
        if self.daily_start_value is None:
            self.daily_start_value = portfolio.total_value
            self.daily_low_value = portfolio.total_value

        # Update daily low
        if portfolio.total_value < self.daily_low_value:
            self.daily_low_value = portfolio.total_value

        # Circuit breaker: Check daily loss limit
        daily_loss_pct = ((portfolio.total_value - self.daily_start_value) /
                         self.daily_start_value * 100)

        if daily_loss_pct <= -self.config.max_daily_loss_pct:
            return RiskStatus(
                approved=False,
                halt=True,
                reason=f"Circuit breaker triggered: Daily loss {daily_loss_pct:.2f}% exceeds limit of {self.config.max_daily_loss_pct}%"
            )

        # Check position-level stop loss
        for symbol, position in portfolio.positions.items():
            loss_pct = ((position.current_price - position.avg_entry_price) /
                       position.avg_entry_price * 100)

            if loss_pct <= -self.config.stop_loss_pct:
                return RiskStatus(
                    approved=False,
                    halt=False,
                    reason=f"Stop loss triggered for {symbol}: {loss_pct:.2f}%"
                )

        return RiskStatus(approved=True)

    def validate_order(self, order: Order, portfolio: Portfolio) -> RiskStatus:
        """Validate order against risk limits"""

        # Check position size limit
        if order.action == TradingAction.BUY:
            if order.shares > self.config.max_position_size:
                return RiskStatus(
                    approved=False,
                    reason=f"Order size {order.shares} exceeds max position size {self.config.max_position_size}"
                )

            # Check if enough cash
            cost = order.shares * order.price + self.config.commission_per_trade
            if cost > portfolio.cash:
                return RiskStatus(
                    approved=False,
                    reason=f"Insufficient cash: Need ${cost:.2f}, have ${portfolio.cash:.2f}"
                )

        elif order.action == TradingAction.SELL:
            # Check if have enough shares
            position = portfolio.positions.get(order.symbol)
            if not position or position.shares < order.shares:
                return RiskStatus(
                    approved=False,
                    reason=f"Insufficient shares: Trying to sell {order.shares}, have {position.shares if position else 0}"
                )

        return RiskStatus(approved=True)

    def reset_daily_tracking(self):
        """Reset daily tracking (call at start of new trading day)"""
        self.daily_start_value = None
        self.daily_low_value = None


# ============================================================================
# Order Executor
# ============================================================================

class OrderExecutor:
    """Simulated order execution"""

    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self._trade_counter = 0

    def validate(self, order: Order, portfolio: Portfolio) -> bool:
        """Validate order can be executed"""
        if order.action == TradingAction.BUY:
            cost = order.shares * order.price + self.config.commission_per_trade
            return cost <= portfolio.cash

        elif order.action == TradingAction.SELL:
            position = portfolio.positions.get(order.symbol)
            return position is not None and position.shares >= order.shares

        return False

    def execute(self, order: Order, portfolio: Portfolio) -> Trade:
        """Execute order and update portfolio"""

        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:06d}"

        if order.action == TradingAction.BUY:
            # Execute buy
            cost = order.shares * order.price
            total_cost = cost + self.config.commission_per_trade

            portfolio.cash -= total_cost

            # Update or create position
            if order.symbol in portfolio.positions:
                pos = portfolio.positions[order.symbol]
                total_shares = pos.shares + order.shares
                total_cost_basis = (pos.avg_entry_price * pos.shares) + cost
                pos.avg_entry_price = total_cost_basis / total_shares
                pos.shares = total_shares
                pos.current_price = order.price
            else:
                portfolio.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    shares=order.shares,
                    avg_entry_price=order.price,
                    current_price=order.price
                )

            trade = Trade(
                symbol=order.symbol,
                action=TradingAction.BUY,
                shares=order.shares,
                price=order.price,
                timestamp=order.timestamp,
                pnl=0.0,
                commission=self.config.commission_per_trade,
                trade_id=trade_id
            )

        elif order.action == TradingAction.SELL:
            # Execute sell
            position = portfolio.positions[order.symbol]

            proceeds = order.shares * order.price
            total_proceeds = proceeds - self.config.commission_per_trade

            # Calculate P&L
            pnl = (order.price - position.avg_entry_price) * order.shares - self.config.commission_per_trade

            portfolio.cash += total_proceeds
            position.shares -= order.shares
            position.realized_pnl += pnl

            # Remove position if fully closed
            if position.shares == 0:
                del portfolio.positions[order.symbol]

            trade = Trade(
                symbol=order.symbol,
                action=TradingAction.SELL,
                shares=order.shares,
                price=order.price,
                timestamp=order.timestamp,
                pnl=pnl,
                commission=self.config.commission_per_trade,
                trade_id=trade_id
            )

        else:
            raise ValueError(f"Invalid action: {order.action}")

        portfolio.trades.append(trade)
        order.status = OrderStatus.EXECUTED

        logger.info(f"Executed {trade.action.name} {trade.shares} {trade.symbol} @ ${trade.price:.2f} (P&L: ${trade.pnl:.2f})")

        return trade


# ============================================================================
# Live Trading Engine
# ============================================================================

class LiveTradingEngine:
    """Main live trading engine orchestrator"""

    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.portfolio = Portfolio(
            initial_cash=config.initial_capital,
            cash=config.initial_capital
        )
        self.market_stream = MarketDataStream(self.config)
        self.risk_manager = RiskManager(config)
        self.order_executor = OrderExecutor(config)
        self.agent = None  # Will be loaded
        self.session: Optional[TradingSession] = None
        self._is_running = False

    def save_state(self, file_path: Path):
        """Save the current trading session state to a file"""
        if not self.session:
            logger.warning("No active session to save.")
            return

        try:
            state = self.session.to_dict()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Successfully saved trading session to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save trading session: {e}")

    @classmethod
    def load_from_state(cls, file_path: Path) -> "LiveTradingEngine":
        """Load a trading session state from a file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            session = TradingSession.from_dict(state)
            
            # Recreate the engine from the loaded session
            engine = cls(session.config)
            engine.session = session
            engine.portfolio = session.portfolio
            
            # Set running status
            engine._is_running = session.status == TradingStatus.RUNNING
            
            logger.info(f"Successfully loaded trading session from {file_path}")
            return engine
        except Exception as e:
            logger.error(f"Failed to load trading session: {e}")
            raise



    def load_agent(self, agent_path: str):
        """Load trained RL agent"""
        from stable_baselines3 import PPO, A2C

        try:
            agent_path = Path(agent_path)
            if not agent_path.exists():
                raise FileNotFoundError(f"Agent not found: {agent_path}")

            # Try to load as PPO first, then A2C
            try:
                self.agent = PPO.load(str(agent_path))
                logger.info(f"Loaded PPO agent from {agent_path}")
            except Exception as e:
                try:
                    self.agent = A2C.load(str(agent_path))
                    logger.info(f"Loaded A2C agent from {agent_path}")
                except Exception as e2:
                    raise Exception(f"Failed to load as PPO or A2C: {e}, {e2}")

        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            raise

    def start_session(self) -> TradingSession:
        """Start new trading session"""
        session_id = f"SESSION_{self.config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session = TradingSession(
            session_id=session_id,
            config=self.config,
            portfolio=self.portfolio,
            status=TradingStatus.RUNNING,
            start_time=datetime.now()
        )

        self._is_running = True
        self.session.add_event("SESSION_START", f"Started live trading session with ${self.config.initial_capital:.2f}")

        logger.info(f"Started trading session: {session_id}")
        return self.session

    def stop_session(self):
        """Stop current trading session"""
        if self.session:
            self.session.status = TradingStatus.STOPPED
            self.session.end_time = datetime.now()
            self.session.add_event("SESSION_END", f"Ended session. Final P&L: ${self.portfolio.total_pnl:.2f}")



            self._is_running = False
            logger.info(f"Stopped trading session: {self.session.session_id}")

    def trading_cycle(self) -> Dict:
        """Single iteration of trading logic"""
        if not self._is_running or not self.session:
            return {"status": "not_running"}

        try:
            # Get latest market data
            tick = self.market_stream.get_latest_tick()
            logger.info(f"Trading cycle: {self.config.symbol} @ ${tick.price:.2f}, Portfolio: ${self.portfolio.total_value:,.2f}")

            # Update portfolio valuations
            self.portfolio.update_valuations(tick)

            # Check trading hours if required
            if self.config.enforce_trading_hours and not self.market_stream.is_market_open():
                logger.info("Market closed - skipping trading cycle")
                return {
                    "status": "market_closed",
                    "timestamp": tick.timestamp,
                    "portfolio_value": self.portfolio.total_value
                }

            # Risk management check
            risk_status = self.risk_manager.check(self.portfolio, tick)

            if risk_status.halt:
                self.session.status = TradingStatus.HALTED
                self.session.add_event("HALT", risk_status.reason)
                self._is_running = False
                return {
                    "status": "halted",
                    "reason": risk_status.reason,
                    "portfolio_value": self.portfolio.total_value
                }

            # Get agent decision
            if self.agent is None:
                return {"status": "no_agent"}

            # Build observation for agent (simplified - would need proper feature engineering)
            observation = self._build_observation(tick)

            # Get action from agent
            action, _states = self.agent.predict(observation, deterministic=True)

            # Map action to trading decision
            if action == 0:  # SELL
                trading_action = TradingAction.SELL
            elif action == 2:  # BUY
                trading_action = TradingAction.BUY
            else:  # HOLD
                trading_action = TradingAction.HOLD

            logger.info(f"Agent decision: {trading_action.name} (action={action})")

            # Execute trade if not HOLD
            if trading_action != TradingAction.HOLD:
                # Determine shares (simplified - could use position sizing logic)
                shares = min(10, self.config.max_position_size)

                order = Order(
                    symbol=self.config.symbol,
                    action=trading_action,
                    shares=shares,
                    price=tick.price,
                    timestamp=tick.timestamp
                )

                # Validate order
                order_risk_status = self.risk_manager.validate_order(order, self.portfolio)

                if order_risk_status.approved and self.order_executor.validate(order, self.portfolio):
                    trade = self.order_executor.execute(order, self.portfolio)
                    self.session.add_event("TRADE", f"{trade.action.name} {trade.shares} @ ${trade.price:.2f}")
                    logger.info(f"✅ Trade executed: {trade.action.name} {trade.shares} shares @ ${trade.price:.2f}, P&L: ${trade.pnl:+.2f}")

                    return {
                        "status": "trade_executed",
                        "trade": trade,
                        "portfolio_value": self.portfolio.total_value,
                        "timestamp": tick.timestamp
                    }
                else:
                    reason = order_risk_status.reason or "Order validation failed"
                    self.session.add_event("ORDER_REJECTED", reason)
                    logger.warning(f"❌ Order rejected: {reason}")
                    return {
                        "status": "order_rejected",
                        "reason": reason,
                        "portfolio_value": self.portfolio.total_value
                    }

            # No action taken
            logger.info("⏸️  HOLD - No trade executed")
            return {
                "status": "hold",
                "portfolio_value": self.portfolio.total_value,
                "timestamp": tick.timestamp,
                "price": tick.price
            }

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.session.add_event("ERROR", str(e))
            return {"status": "error", "message": str(e)}

    def _build_observation(self, tick: MarketTick) -> np.ndarray:
        """Build observation matching training environment format (60, 10)"""
        from ..tools.stock_fetcher import StockFetcher
        from ..tools.technical_analysis import TechnicalAnalysis
        from datetime import datetime, timedelta

        try:
            # Fetch 90 days of historical data to ensure we have enough after calculating indicators
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

            fetcher = StockFetcher()
            hist_data = fetcher.fetch_stock_data(
                symbol=self.config.symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d',
                force_refresh=True  # Always fetch fresh data for live trading
            )

            if hist_data is None or len(hist_data) < 60:
                logger.warning(f"Insufficient historical data for observation, using zeros")
                return np.zeros((60, 10), dtype=np.float32)

            # Calculate technical indicators
            close = hist_data['Close']
            high = hist_data['High']
            low = hist_data['Low']

            # RSI
            rsi = TechnicalAnalysis.calculate_rsi(close)

            # MACD
            macd_indicators = TechnicalAnalysis.calculate_macd(close)
            macd = macd_indicators['macd']
            macd_signal = macd_indicators['macd_signal']

            # Bollinger Bands
            bb_indicators = TechnicalAnalysis.calculate_bollinger_bands(close)
            bb_upper = bb_indicators['bb_upper']
            bb_lower = bb_indicators['bb_lower']

            # Stochastic
            stoch_indicators = TechnicalAnalysis.calculate_stochastic(high, low, close)
            stochastic = stoch_indicators['stoch_k']

            # Fill NaN values
            hist_data = hist_data.bfill().ffill()
            rsi = rsi.bfill().ffill()
            macd = macd.bfill().ffill()
            macd_signal = macd_signal.bfill().ffill()
            bb_upper = bb_upper.bfill().ffill()
            bb_lower = bb_lower.bfill().ffill()
            stochastic = stochastic.bfill().ffill()

            # Get last 60 days
            close_last_60 = close.iloc[-60:].values
            volume_last_60 = hist_data['Volume'].iloc[-60:].values
            rsi_last_60 = rsi.iloc[-60:].values
            macd_last_60 = macd.iloc[-60:].values
            macd_signal_last_60 = macd_signal.iloc[-60:].values
            bb_upper_last_60 = bb_upper.iloc[-60:].values
            bb_lower_last_60 = bb_lower.iloc[-60:].values
            stochastic_last_60 = stochastic.iloc[-60:].values

            # Normalize features
            first_price = close_last_60[0]
            close_norm = (close_last_60 - first_price) / first_price

            max_volume = volume_last_60.max()
            volume_norm = volume_last_60 / (max_volume + 1e-8)

            rsi_norm = rsi_last_60 / 100.0
            macd_norm = macd_last_60 / (close_last_60 + 1e-8)
            macd_signal_norm = macd_signal_last_60 / (close_last_60 + 1e-8)
            bb_position = (close_last_60 - bb_lower_last_60) / (bb_upper_last_60 - bb_lower_last_60 + 1e-8)
            stochastic_norm = stochastic_last_60 / 100.0

            # Portfolio state (repeated for all timesteps)
            position = self.portfolio.positions.get(self.config.symbol)
            position_shares = position.shares if position else 0
            current_price = tick.price
            portfolio_value = self.portfolio.total_value

            cash_ratio = np.full(60, self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0)
            position_ratio = np.full(60, (position_shares * current_price) / portfolio_value if portfolio_value > 0 else 0.0)

            # Portfolio value change
            prev_value = getattr(self, '_prev_portfolio_value', self.config.initial_capital)
            value_change = np.full(60, (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0)
            self._prev_portfolio_value = portfolio_value

            # Stack all features (shape: 60 x 10)
            observation = np.column_stack([
                close_norm,
                volume_norm,
                cash_ratio,
                position_ratio,
                value_change,
                rsi_norm,
                macd_norm,
                macd_signal_norm,
                bb_position,
                stochastic_norm
            ]).astype(np.float32)

            return observation

        except Exception as e:
            logger.error(f"Error building observation: {e}")
            # Return zeros as fallback
            return np.zeros((60, 10), dtype=np.float32)
