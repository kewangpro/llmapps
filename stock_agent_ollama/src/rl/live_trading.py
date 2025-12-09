"""
Live Trading Simulation System

Educational paper trading system that uses trained RL models with real-time market data.
This is for SIMULATION ONLY - no real money is involved.
"""

from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from src.rl.environments import TradingAction as EnvTradingAction
from src.rl.env_factory import EnvConfig
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Reference defaults from EnvConfig (single source of truth)
_ENV_DEFAULTS = {f.name: f.default for f in EnvConfig.__dataclass_fields__.values()}


# ============================================================================
# Enums and Constants
# ============================================================================

# Use the environment's TradingAction IntEnum for consistency with training
# EnvTradingAction values: SELL=0, HOLD=1, BUY_SMALL=2, BUY_LARGE=3
class TradingAction(Enum):
    """Compatibility wrapper mapping to environment TradingAction values.

    Keep a small Enum here for typing and backwards compatibility in other
    parts of the code (UI expects BUY/SELL/HOLD labels). Internally we map
    to the environment's action integers.
    """
    SELL = int(EnvTradingAction.SELL)
    HOLD = int(EnvTradingAction.HOLD)
    BUY_SMALL = int(EnvTradingAction.BUY_SMALL)
    BUY_LARGE = int(EnvTradingAction.BUY_LARGE)


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
    initial_capital: float = _ENV_DEFAULTS['initial_balance']
    max_position_size: float = _ENV_DEFAULTS['max_position_pct']  # Max position as % of portfolio value (from EnvConfig)
    max_portfolio_risk_pct: float = 2.0  # Max 2% portfolio risk per trade
    stop_loss_pct: float = 5.0  # 5% stop loss
    max_daily_loss_pct: float = 10.0  # 10% max daily loss (circuit breaker)
    update_interval: int = 60  # Seconds between updates
    enforce_trading_hours: bool = True
    auto_select_stock: bool = False  # Dynamically select stocks to maximize returns
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
    # New fields for multi-session support
    strategy_name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    color: str = "#7C3AED"
    display_order: int = 0
    # Track last stock rotation time (for auto-select cooldown)
    last_rotation_time: Optional[datetime] = None
    cycles_since_rotation: int = 0

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
            "strategy_name": self.strategy_name,
            "description": self.description,
            "tags": self.tags,
            "color": self.color,
            "display_order": self.display_order,
            "last_rotation_time": self.last_rotation_time.isoformat() if self.last_rotation_time else None,
            "cycles_since_rotation": self.cycles_since_rotation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSession":
        last_rotation_time = data.get("last_rotation_time")
        return cls(
            session_id=data["session_id"],
            config=LiveTradingConfig.from_dict(data["config"]),
            portfolio=Portfolio.from_dict(data["portfolio"]),
            status=TradingStatus(data["status"]),
            start_time=datetime.fromisoformat(data["start_time"]) if data["start_time"] else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            events=data.get("events", []),
            strategy_name=data.get("strategy_name", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            color=data.get("color", "#7C3AED"),
            display_order=data.get("display_order", 0),
            last_rotation_time=datetime.fromisoformat(last_rotation_time) if last_rotation_time else None,
            cycles_since_rotation=data.get("cycles_since_rotation", 0)
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
        """Check if market is currently open (regular hours only)"""
        try:
            # Create a fresh ticker to get the most up-to-date info
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            market_state = info.get('marketState', 'CLOSED').upper()

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

                return market_open

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
        # Check position size limit (as % of portfolio value)
        if order.action in (TradingAction.BUY_SMALL, TradingAction.BUY_LARGE):
            # Calculate max shares allowed based on portfolio value percentage
            max_position_value = portfolio.total_value * (self.config.max_position_size / 100.0)
            max_shares = int(max_position_value / order.price)

            # Get current position
            current_position = portfolio.positions.get(order.symbol)
            current_shares = current_position.shares if current_position else 0

            # Check if adding these shares would exceed limit
            total_shares_after = current_shares + order.shares
            total_value_after = total_shares_after * order.price
            position_pct = (total_value_after / portfolio.total_value) * 100

            # Use small tolerance for floating point comparison (allow up to max + 0.01%)
            if position_pct > self.config.max_position_size + 0.01:
                return RiskStatus(
                    approved=False,
                    reason=f"Position would be {position_pct:.1f}% of portfolio, max is {self.config.max_position_size:.1f}%"
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
        if order.action in (TradingAction.BUY_SMALL, TradingAction.BUY_LARGE):
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

        if order.action in (TradingAction.BUY_SMALL, TradingAction.BUY_LARGE):
            # Execute buy (support small/large buys)
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
                action=order.action,
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
                action=order.action,
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
        self.env = None  # Will be initialized when needed

    def _add_event(self, event_type: str, message: str):
        """Add event to session log with symbol prefix for auto-select sessions"""
        if self.config.auto_select_stock:
            # Prefix message with symbol for auto-select sessions
            message = f"[{self.config.symbol}] {message}"
        self.session.add_event(event_type, message)

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
            logger.info(f"{self.session.session_id} - Successfully saved trading session to {file_path}")
        except Exception as e:
            logger.error(f"{self.session.session_id} - Failed to save trading session: {e}")

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
        """Load trained RL agent using centralized model loader"""
        from .model_utils import load_rl_agent

        try:
            agent_path = Path(agent_path)
            if not agent_path.exists():
                raise FileNotFoundError(f"Agent not found: {agent_path}")

            # Use centralized loader that handles PPO, RecurrentPPO, SAC, QRDQN
            self.agent = load_rl_agent(agent_path, env=None)
            logger.info(f"Successfully loaded agent from {agent_path}")

        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            raise

    def setup_environment(self):
        """
        Setup enhanced trading environment matching training configuration.

        This loads the complete environment config from the trained model to ensure
        live trading uses the exact same settings as training.
        """
        from .model_utils import load_env_config_from_model
        from .env_factory import EnvConfig, create_enhanced_env
        from .improvements import EnhancedRewardConfig
        from datetime import datetime, timedelta

        # Load config from trained model to match training environment
        try:
            training_config = load_env_config_from_model(Path(self.config.agent_path))
            logger.info(f"Loaded training config from model: {Path(self.config.agent_path).parent}")
        except Exception as e:
            logger.warning(f"Could not load training config ({e}), using default values")
            training_config = {}

        # Use recent historical data for environment (override saved dates)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        logger.info(f"Initializing environment with data from {start_date} to {end_date}")

        # Detect feature configuration from agent's observation space
        include_trend_indicators = False
        if self.agent is not None and hasattr(self.agent, 'observation_space'):
            obs_shape = self.agent.observation_space.shape
            lookback_window = obs_shape[0] if len(obs_shape) > 0 else 60
            expected_features = obs_shape[1] if len(obs_shape) > 1 else 10

            # Infer include_trend_indicators from feature count
            # Base: 5, Tech: 5, Trend: 3, Regime: 7, MTF: 6
            # With all enabled: 5 + 5 + 3 + 7 + 6 = 26
            # Without trend: 5 + 5 + 7 + 6 = 23
            use_regime = training_config.get('use_regime_detector', True)
            use_mtf = training_config.get('use_mtf_features', True)

            base_features = 5 + 5  # base + technical
            if use_regime:
                base_features += 7
            if use_mtf:
                base_features += 6

            # If we have 3 more features than expected, trend indicators are enabled
            if expected_features == base_features + 3:
                include_trend_indicators = True
                logger.info(f"Detected include_trend_indicators=True from observation shape {obs_shape}")
            else:
                logger.info(f"Detected include_trend_indicators=False from observation shape {obs_shape}")
        else:
            # Fall back to config
            include_trend_indicators = training_config.get('include_trend_indicators', False)

        # Build env config matching training, with live date ranges
        env_config = EnvConfig(
            symbol=self.config.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.config.initial_capital,

            # Load critical parameters from training config
            transaction_cost_rate=training_config.get('transaction_cost_rate', 0.0005),
            slippage_rate=training_config.get('slippage_rate', 0.0005),
            max_position_size=training_config.get('max_position_size', 1000),
            max_position_pct=training_config.get('max_position_pct', 80.0),
            lookback_window=training_config.get('lookback_window', 60),
            include_technical_indicators=training_config.get('include_technical_indicators', True),
            include_trend_indicators=include_trend_indicators,  # Use detected value

            # Load enhancement flags from training config
            use_action_masking=training_config.get('use_action_masking', True),
            use_enhanced_rewards=training_config.get('use_enhanced_rewards', True),
            use_adaptive_sizing=training_config.get('use_adaptive_sizing', True),
            use_improved_actions=training_config.get('use_improved_actions', True),
            use_regime_detector=training_config.get('use_regime_detector', True),
            use_mtf_features=training_config.get('use_mtf_features', True),

            # No curriculum or diagnostics in live trading
            curriculum_manager=None,
            enable_diagnostics=False,

            # Use default reward config
            reward_config=EnhancedRewardConfig()
        )

        # Create environment using shared factory
        self.env = create_enhanced_env(env_config)

        logger.info(f"Environment initialized for live trading with training config: "
                   f"costs={env_config.transaction_cost_rate:.4f}, "
                   f"max_pos={env_config.max_position_pct}%, "
                   f"obs_shape={self.env.observation_space.shape}")

    def initialize_session_state(self, session_id: Optional[str] = None) -> TradingSession:
        """Initialize new trading session state"""
        if session_id is None:
            # Differentiate auto-select sessions with SESSION_AUTO_ prefix
            if self.config.auto_select_stock:
                session_id = f"SESSION_AUTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                session_id = f"SESSION_{self.config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session = TradingSession(
            session_id=session_id,
            config=self.config,
            portfolio=self.portfolio,
            status=TradingStatus.IDLE, # Start as IDLE
            start_time=None # Set start_time when actually started
        )

        logger.info(f"{session_id} - Initialized trading session state")
        return self.session

    def stop_session(self):
        """Stop current trading session"""
        if self.session:
            self.session.status = TradingStatus.STOPPED
            self.session.end_time = datetime.now()
            self._add_event("SESSION_END", f"Ended session. Final P&L: ${self.portfolio.total_pnl:.2f}")



            self._is_running = False
            logger.info(f"{self.session.session_id} - Stopped trading session")

    def _evaluate_stock_performance(self, symbol: str, days: int = 5) -> Optional[float]:
        """Evaluate recent stock performance (return %)"""
        try:
            from ..tools.stock_fetcher import StockFetcher
            from datetime import datetime, timedelta

            fetcher = StockFetcher()
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")  # Extra buffer

            data = fetcher.fetch_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d',
                force_refresh=True
            )

            if data is None or len(data) < 2:
                return None

            # Calculate recent return (last N days)
            recent_data = data.tail(min(days, len(data)))
            if len(recent_data) < 2:
                return None

            start_price = recent_data.iloc[0]['Close']
            end_price = recent_data.iloc[-1]['Close']
            return_pct = ((end_price - start_price) / start_price) * 100

            return return_pct

        except Exception as e:
            logger.warning(f"Failed to evaluate {symbol} performance: {e}")
            return None

    def _find_best_model_for_symbol(self, symbol: str) -> Optional[Tuple[str, str]]:
        """Find best performing model (algorithm + path) for a symbol using intelligent heuristics"""
        from pathlib import Path
        import json
        from datetime import datetime

        models_dir = Path("data/models/rl")
        if not models_dir.exists():
            return None

        # Collect all available models with scoring
        model_candidates = []

        for agent_type in ['ensemble', 'recurrent_ppo', 'ppo']:
            agent_type_lower = agent_type.lower()
            pattern = f"{agent_type_lower}_{symbol}_*"
            matching_dirs = list(models_dir.glob(pattern))

            if matching_dirs:
                # Sort by modification time (most recent first)
                matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                latest = matching_dirs[0]

                # Calculate composite score based on multiple factors
                score = 0.0

                # Factor 1: Algorithm preference (RecurrentPPO > PPO > Ensemble)
                # Based on typical performance, RecurrentPPO often performs best
                algo_scores = {
                    'recurrent_ppo': 100,  # Prefer RecurrentPPO (LSTM memory)
                    'ppo': 80,              # PPO is solid baseline
                    'ensemble': 60          # Ensemble can be inconsistent
                }
                score += algo_scores.get(agent_type_lower, 0)

                # Factor 2: Model recency (newer models may have better training)
                model_age_days = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).days
                recency_score = max(0, 20 - model_age_days)  # Up to 20 points for recent models
                score += recency_score

                # Factor 3: Training quality (more timesteps = better training)
                training_config_file = latest / "training_config.json"
                if training_config_file.exists():
                    try:
                        with open(training_config_file, 'r') as f:
                            config = json.load(f)
                            timesteps = config.get('total_timesteps', 0)
                            # Give points for longer training (up to 20 points)
                            if timesteps >= 300000:
                                score += 20
                            elif timesteps >= 100000:
                                score += 10
                            elif timesteps >= 50000:
                                score += 5
                    except Exception as e:
                        logger.debug(f"Could not load training config for {latest.name}: {e}")

                model_candidates.append({
                    'agent_type': agent_type,
                    'path': str(latest),
                    'score': score,
                    'age_days': model_age_days
                })

        if not model_candidates:
            return None

        # Select model with highest composite score
        best_candidate = max(model_candidates, key=lambda x: x['score'])
        logger.info(f"Selected {best_candidate['agent_type']} for {symbol} (score: {best_candidate['score']:.0f}, age: {best_candidate['age_days']}d)")

        return (best_candidate['agent_type'], best_candidate['path'])

    def _select_best_stock(self) -> Optional[Tuple[str, str]]:
        """Select best stock from watchlist based on recent performance
        Returns: (symbol, agent_path) or None
        """
        try:
            from ..tools.portfolio_manager import portfolio_manager

            # Minimum performance improvement required to justify rotation (2%)
            MIN_PERFORMANCE_IMPROVEMENT_PCT = 2.0

            watchlist = portfolio_manager.load_portfolio("default")
            if not watchlist:
                logger.warning("No stocks in watchlist for auto-selection")
                return None

            # Evaluate all watchlist stocks (including current symbol)
            stock_scores = []
            current_symbol_performance = None

            for symbol in watchlist:
                # Check if model exists
                model_info = self._find_best_model_for_symbol(symbol)
                if not model_info:
                    continue

                # Evaluate recent performance
                performance = self._evaluate_stock_performance(symbol, days=5)
                if performance is None:
                    continue

                stock_scores.append({
                    'symbol': symbol,
                    'algorithm': model_info[0],
                    'agent_path': model_info[1],
                    'performance': performance
                })

                # Track current symbol's performance
                if symbol == self.config.symbol:
                    current_symbol_performance = performance

            if not stock_scores:
                logger.warning("No suitable stocks found for auto-selection")
                return None

            # Sort by performance (descending)
            stock_scores.sort(key=lambda x: x['performance'], reverse=True)
            best_stock = stock_scores[0]

            # Only rotate if the best stock is significantly better than current stock
            if current_symbol_performance is not None:
                performance_diff = best_stock['performance'] - current_symbol_performance

                if best_stock['symbol'] == self.config.symbol:
                    # Current stock is still the best, no rotation needed
                    logger.info(f"Auto-select: {self.config.symbol} remains the best with {current_symbol_performance:.2f}% recent return")
                    return None
                elif performance_diff < MIN_PERFORMANCE_IMPROVEMENT_PCT:
                    # Not enough improvement to justify rotation
                    logger.info(f"Auto-select: {best_stock['symbol']} is only {performance_diff:.2f}% better than {self.config.symbol} (threshold: {MIN_PERFORMANCE_IMPROVEMENT_PCT}%), staying with current stock")
                    return None

            logger.info(f"Auto-select: {best_stock['symbol']} ({best_stock['algorithm']}) with {best_stock['performance']:.2f}% recent return")

            return (best_stock['symbol'], best_stock['agent_path'])

        except Exception as e:
            logger.error(f"Error selecting best stock: {e}")
            return None

    def _rotate_to_stock(self, symbol: str, agent_path: str):
        """Rotate to a new stock by updating config and reloading agent"""
        try:
            logger.info(f"{self.session.session_id} - Rotating from {self.config.symbol} to {symbol}")

            # Extract model name from agent_path
            from pathlib import Path
            agent_path_obj = Path(agent_path)
            # Handle different path formats
            if agent_path_obj.is_file() or agent_path_obj.name in ['ensemble', 'ppo', 'recurrent_ppo']:
                model_name = agent_path_obj.parent.name
            else:
                model_name = agent_path_obj.name

            # Update config
            self.config.symbol = symbol
            self.config.agent_path = agent_path

            # Reload agent
            self.load_agent(agent_path)

            # Reset environment (will be re-initialized on next cycle)
            self.env = None

            # Reinitialize market stream
            self.market_stream = MarketDataStream(self.config)

            # Update rotation tracking
            self.session.last_rotation_time = datetime.now()
            self.session.cycles_since_rotation = 0

            # Add event with model name (don't use _add_event here as we want the NEW symbol shown)
            self.session.add_event("STOCK_ROTATION", f"Rotated to {symbol} ({model_name})")

            logger.info(f"{self.session.session_id} - Successfully rotated to {symbol} with {model_name}")

        except Exception as e:
            logger.error(f"Failed to rotate to {symbol}: {e}")
            self._add_event("ROTATION_FAILED", f"Failed to rotate to {symbol}: {str(e)}")

    def trading_cycle(self) -> Dict:
        """Single iteration of trading logic"""
        if not self._is_running or not self.session:
            return {"status": "not_running"}

        try:
            # Get latest market data
            tick = self.market_stream.get_latest_tick()
            logger.info(f"{self.session.session_id} - Trading cycle: {self.config.symbol} @ ${tick.price:.2f}, Portfolio: ${self.portfolio.total_value:,.2f}")

            # Update portfolio valuations
            self.portfolio.update_valuations(tick)

            # Check trading hours if required
            if self.config.enforce_trading_hours and not self.market_stream.is_market_open():
                logger.info(f"{self.session.session_id} - Market closed - skipping trading cycle")
                return {
                    "status": "market_closed",
                    "timestamp": tick.timestamp.isoformat(),
                    "portfolio_value": self.portfolio.total_value
                }

            # Risk management check
            risk_status = self.risk_manager.check(self.portfolio, tick)

            if risk_status.halt:
                self.session.status = TradingStatus.HALTED
                self._add_event("HALT", risk_status.reason)
                self._is_running = False
                logger.error(f"{self.session.session_id} - {risk_status.reason}")
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

            # Initialize environment if not already done
            if self.env is None:
                self.setup_environment()

            # Use ImprovedTradingAction enum (6 actions)
            from .improvements import ImprovedTradingAction

            # Get current position for action masking
            current_position = self.portfolio.positions.get(self.config.symbol)
            current_shares = current_position.shares if current_position else 0

            # Auto-select stock rotation: if enabled and position is 0, consider rotating to better stock
            # Only rotate after giving current stock enough time (minimum 10 cycles or 10 minutes)
            MIN_CYCLES_BEFORE_ROTATION = 10
            MIN_MINUTES_BEFORE_ROTATION = 10

            if self.config.auto_select_stock and current_shares == 0:
                # Increment cycle counter
                self.session.cycles_since_rotation += 1

                # Check if enough time has passed
                cycles_elapsed = self.session.cycles_since_rotation
                time_elapsed_minutes = 0
                if self.session.last_rotation_time:
                    time_elapsed_minutes = (datetime.now() - self.session.last_rotation_time).total_seconds() / 60

                can_rotate = (cycles_elapsed >= MIN_CYCLES_BEFORE_ROTATION or
                             time_elapsed_minutes >= MIN_MINUTES_BEFORE_ROTATION)

                if can_rotate:
                    logger.info(f"{self.session.session_id} - Auto-select mode: Position is 0 for {cycles_elapsed} cycles, checking for better stock...")
                    best_stock = self._select_best_stock()

                    if best_stock:
                        new_symbol, new_agent_path = best_stock
                        if new_symbol != self.config.symbol:
                            # Rotate to new stock
                            self._rotate_to_stock(new_symbol, new_agent_path)

                            # Return early to allow next cycle to trade the new stock
                            return {
                                "status": "stock_rotated",
                                "new_symbol": new_symbol,
                                "portfolio_value": self.portfolio.total_value,
                                "timestamp": tick.timestamp.isoformat()
                            }
                else:
                    logger.debug(f"{self.session.session_id} - Auto-select: Position is 0 but only {cycles_elapsed} cycles since rotation, waiting...")
            else:
                # Reset counter if we have a position
                if current_shares > 0:
                    self.session.cycles_since_rotation = 0

            # Get action mask from environment
            action_mask = self.env.action_masker.get_action_mask(
                cash=self.portfolio.cash,
                position=current_shares,
                current_price=tick.price,
                portfolio_value=self.portfolio.total_value,
                max_position_pct=self.config.max_position_size
            )

            # Get action from agent
            action, _states = self.agent.predict(observation, deterministic=True)

            action_names = {
                0: 'HOLD',
                1: 'BUY_SMALL',
                2: 'BUY_MEDIUM',
                3: 'BUY_LARGE',
                4: 'SELL_PARTIAL',
                5: 'SELL_ALL'
            }

            action_name = action_names.get(int(action), 'UNKNOWN')

            # Debug logging - show action mask details
            logger.info(f"{self.session.session_id} - Action mask: {action_mask}")
            logger.info(f"{self.session.session_id} - Portfolio: Cash=${self.portfolio.cash:.2f}, Position={current_shares} shares, Total Value=${self.portfolio.total_value:.2f}, Price=${tick.price:.2f}")

            # Check if predicted action is valid according to mask
            if action_mask[int(action)] == 0.0:
                logger.warning(f"{self.session.session_id} - Agent predicted invalid action: {action_name}, defaulting to HOLD")
                self._add_event("ACTION_MASKED", f"Agent predicted {action_name} but action is invalid, using HOLD")
                action = ImprovedTradingAction.HOLD
                action_name = 'HOLD'

            logger.info(f"{self.session.session_id} - Agent decision: {action_name} (action={action})")

            # Execute trade if not HOLD (action 0)
            if int(action) != ImprovedTradingAction.HOLD:
                # Calculate shares using adaptive sizing from environment
                shares = 0

                if int(action) in [ImprovedTradingAction.BUY_SMALL, ImprovedTradingAction.BUY_MEDIUM, ImprovedTradingAction.BUY_LARGE]:
                    # Use environment's adaptive sizer
                    if self.env and self.env.adaptive_sizer:
                        shares = self.env.adaptive_sizer.get_buy_size(
                            action=int(action),
                            cash=self.portfolio.cash,
                            price=tick.price,
                            position=current_shares,
                            portfolio_value=self.portfolio.total_value,
                            max_position_pct=self.config.max_position_size,
                            volatility=0.02,  # Could calculate from recent data
                            use_improved_actions=True
                        )

                elif int(action) == ImprovedTradingAction.SELL_PARTIAL:
                    # Sell 50% of position (use environment's logic)
                    if self.env and self.env.adaptive_sizer:
                        sell_shares = self.env.adaptive_sizer.get_sell_size(
                            action=int(action),
                            position=current_shares,
                            use_improved_actions=True
                        )
                        shares = -sell_shares
                    else:
                        shares = -max(1, current_shares // 2)

                elif int(action) == ImprovedTradingAction.SELL_ALL:
                    # Sell all (use environment's logic)
                    if self.env and self.env.adaptive_sizer:
                        sell_shares = self.env.adaptive_sizer.get_sell_size(
                            action=int(action),
                            position=current_shares,
                            use_improved_actions=True
                        )
                        shares = -sell_shares
                    else:
                        shares = -current_shares

                # Skip if no shares to trade
                if shares == 0:
                    reason = f"Calculated trade size is 0 shares for {action_name}"
                    self._add_event("ORDER_SKIPPED", reason)
                    logger.info(f"{self.session.session_id} - ⚠️ Order skipped: {reason}")
                    return {
                        "status": "no_trade",
                        "reason": reason,
                        "portfolio_value": self.portfolio.total_value,
                        "timestamp": tick.timestamp.isoformat()
                    }

                # Convert to TradingAction for execution
                if shares > 0:
                    order_action = TradingAction.BUY_SMALL
                else:
                    order_action = TradingAction.SELL
                    shares = abs(shares)  # Make shares positive for order

                order = Order(
                    symbol=self.config.symbol,
                    action=order_action,
                    shares=shares,
                    price=tick.price,
                    timestamp=tick.timestamp
                )

                # Validate order
                order_risk_status = self.risk_manager.validate_order(order, self.portfolio)

                if order_risk_status.approved and self.order_executor.validate(order, self.portfolio):
                    trade = self.order_executor.execute(order, self.portfolio)
                    self._add_event("TRADE", f"{trade.action.name} {trade.shares} @ ${trade.price:.2f}")
                    logger.info(f"{self.session.session_id} - ✅ Trade executed: {trade.action.name} {trade.shares} shares @ ${trade.price:.2f}, P&L: ${trade.pnl:+.2f}")

                    return {
                        "status": "trade_executed",
                        "trade": trade,
                        "portfolio_value": self.portfolio.total_value,
                        "timestamp": tick.timestamp.isoformat()
                    }
                else:
                    reason = order_risk_status.reason or "Order validation failed"
                    self._add_event("ORDER_REJECTED", reason)
                    logger.warning(f"{self.session.session_id} - ❌ Order rejected: {reason}")
                    return {
                        "status": "order_rejected",
                        "reason": reason,
                        "portfolio_value": self.portfolio.total_value
                    }

            # No action taken
            logger.info(f"{self.session.session_id} - ⏸️  HOLD - No trade executed")
            return {
                "status": "hold",
                "portfolio_value": self.portfolio.total_value,
                "timestamp": tick.timestamp.isoformat(),
                "price": tick.price
            }

        except Exception as e:
            logger.error(f"{self.session.session_id} - Error in trading cycle: {e}")
            self._add_event("ERROR", str(e))
            return {"status": "error", "message": str(e)}

    def _build_observation(self, tick: MarketTick) -> np.ndarray:
        """Build observation matching training environment format (60, 10 or 60, 13)"""
        from ..tools.stock_fetcher import StockFetcher
        from ..tools.technical_analysis import TechnicalAnalysis
        from datetime import datetime, timedelta

        try:
            # Initialize environment if not already done to access config
            if self.env is None:
                self.setup_environment()

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

            # Calculate expected features based on environment config
            expected_features = 5  # base features
            if self.env.include_technical_indicators:
                expected_features += 5  # technical indicators
            if self.env.include_trend_indicators:
                expected_features += 3  # trend indicators
            if getattr(self.env, 'use_regime_detector', False):
                expected_features += 7  # regime features
            if getattr(self.env, 'use_mtf_features', False):
                expected_features += 6  # MTF features

            if hist_data is None or len(hist_data) < 60:
                logger.warning(f"{self.session.session_id} - Insufficient historical data for observation, using zeros")
                return np.zeros((60, expected_features), dtype=np.float32)

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

            # --- Trend Indicators (for LSTM models) ---
            if self.env.include_trend_indicators:
                sma_20 = TechnicalAnalysis.calculate_sma(close, 20)
                ema_12 = TechnicalAnalysis.calculate_ema(close, 12)
                ema_26 = TechnicalAnalysis.calculate_ema(close, 26)

                sma_trend = sma_20.diff(periods=5) / (sma_20.shift(5) + 1e-8)
                ema_crossover = (ema_12 - ema_26) / (ema_26 + 1e-8)
                price_momentum = close.pct_change(periods=5)

            # Fill NaN values
            hist_data = hist_data.bfill().ffill()
            rsi = rsi.bfill().ffill()
            macd = macd.bfill().ffill()
            macd_signal = macd_signal.bfill().ffill()
            bb_upper = bb_upper.bfill().ffill()
            bb_lower = bb_lower.bfill().ffill()
            stochastic = stochastic.bfill().ffill()
            if self.env.include_trend_indicators:
                sma_trend = sma_trend.bfill().ffill()
                ema_crossover = ema_crossover.bfill().ffill()
                price_momentum = price_momentum.bfill().ffill()

            # Get last 60 days
            close_last_60 = close.iloc[-60:].values
            volume_last_60 = hist_data['Volume'].iloc[-60:].values
            rsi_last_60 = rsi.iloc[-60:].values
            macd_last_60 = macd.iloc[-60:].values
            macd_signal_last_60 = macd_signal.iloc[-60:].values
            bb_upper_last_60 = bb_upper.iloc[-60:].values
            bb_lower_last_60 = bb_lower.iloc[-60:].values
            stochastic_last_60 = stochastic.iloc[-60:].values
            if self.env.include_trend_indicators:
                sma_trend_last_60 = sma_trend.iloc[-60:].values
                ema_crossover_last_60 = ema_crossover.iloc[-60:].values
                price_momentum_last_60 = price_momentum.iloc[-60:].values

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

            # Stack all features
            features = [
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
            ]

            if self.env.include_trend_indicators:
                sma_trend_norm = np.clip(sma_trend_last_60, -0.1, 0.1) * 10
                ema_crossover_norm = np.clip(ema_crossover_last_60, -0.2, 0.2) * 5
                price_momentum_norm = np.clip(price_momentum_last_60, -0.1, 0.1) * 10
                features.extend([sma_trend_norm, ema_crossover_norm, price_momentum_norm])

            # --- Regime Features ---
            if getattr(self.env, 'use_regime_detector', False):
                from ..rl.improvements import RegimeDetector
                detector = RegimeDetector()
                regime_features_list = []
                
                # We need to loop over the last 60 steps
                full_close = hist_data['Close'].values
                full_volume = hist_data['Volume'].values if 'Volume' in hist_data.columns else None
                start_idx_in_hist = len(full_close) - 60
                
                for i in range(60):
                    current_slice_idx = start_idx_in_hist + i
                    # Need at least 50 points for detection
                    slice_start = max(0, current_slice_idx - 50)
                    slice_end = current_slice_idx + 1
                    
                    p_slice = full_close[slice_start:slice_end]
                    v_slice = full_volume[slice_start:slice_end] if full_volume is not None else None
                    
                    _, feats = detector.detect_regime(p_slice, v_slice)
                    
                    # Extract 7 features
                    regime_vec = [
                        float(feats['regime_one_hot'][0]),
                        float(feats['regime_one_hot'][1]),
                        float(feats['regime_one_hot'][2]),
                        float(feats['regime_one_hot'][3]),
                        float(feats['trend_strength']),
                        float(feats['trend_direction']),
                        float(feats['volatility_regime'])
                    ]
                    regime_features_list.append(regime_vec)
                
                regime_features_array = np.array(regime_features_list, dtype=np.float32)
                for feat_idx in range(7):
                    features.append(regime_features_array[:, feat_idx])

            # --- MTF Features ---
            if getattr(self.env, 'use_mtf_features', False):
                from ..rl.improvements import MultiTimeframeFeatures
                mtf_extractor = MultiTimeframeFeatures()
                mtf_features_list = []
                
                full_close = hist_data['Close'].values
                full_volume = hist_data['Volume'].values if 'Volume' in hist_data.columns else None
                start_idx_in_hist = len(full_close) - 60
                
                for i in range(60):
                    current_slice_idx = start_idx_in_hist + i
                    slice_start = max(0, current_slice_idx - 50)
                    slice_end = current_slice_idx + 1
                    
                    p_slice = full_close[slice_start:slice_end]
                    v_slice = full_volume[slice_start:slice_end] if full_volume is not None else None
                    
                    feats = mtf_extractor.extract_features(p_slice, v_slice)
                    mtf_features_list.append([float(f) for f in feats])
                    
                mtf_features_array = np.array(mtf_features_list, dtype=np.float32)
                for feat_idx in range(6):
                    features.append(mtf_features_array[:, feat_idx])

            observation = np.column_stack(features).astype(np.float32)

            return observation

        except Exception as e:
            logger.error(f"{self.session.session_id} - Error building observation: {e}")
            # Return zeros as fallback - calculate expected features
            expected_features = 5  # base features
            if hasattr(self, 'env') and self.env:
                if self.env.include_technical_indicators:
                    expected_features += 5
                if self.env.include_trend_indicators:
                    expected_features += 3
                if getattr(self.env, 'use_regime_detector', False):
                    expected_features += 7
                if getattr(self.env, 'use_mtf_features', False):
                    expected_features += 6
            return np.zeros((60, expected_features), dtype=np.float32)
