"""
Live Trading Simulation System

Educational paper trading system that uses trained RL models with real-time market data.
This is for SIMULATION ONLY - no real money is involved.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from src.rl.environments import TradingAction as EnvTradingAction
from src.rl.env_factory import EnvConfig
import numpy as np
import pandas as pd
from src.tools.stock_fetcher import StockFetcher
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
    # Percentage-based transaction costs (matching backtesting)
    transaction_cost_rate: float = _ENV_DEFAULTS['transaction_cost_rate']  # $0 commissions (zero-commission era)
    slippage_rate: float = _ENV_DEFAULTS['slippage_rate']  # 0.05% slippage for liquid stocks
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiveTradingConfig":
        return cls(**data)


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
    # Track consecutive idle cycles (agent not trading)
    consecutive_idle_cycles: int = 0
    # Track recent rotations to prevent ping-pong (symbol -> timestamp left)
    recent_rotations: Dict[str, datetime] = field(default_factory=dict)

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
            "cycles_since_rotation": self.cycles_since_rotation,
            "consecutive_idle_cycles": self.consecutive_idle_cycles,
            "recent_rotations": {k: v.isoformat() for k, v in self.recent_rotations.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSession":
        last_rotation_time = data.get("last_rotation_time")
        recent_rotations_data = data.get("recent_rotations", {})
        recent_rotations = {k: datetime.fromisoformat(v) for k, v in recent_rotations_data.items()}
        
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
            cycles_since_rotation=data.get("cycles_since_rotation", 0),
            consecutive_idle_cycles=data.get("consecutive_idle_cycles", 0),
            recent_rotations=recent_rotations
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
        self.fetcher = StockFetcher()

    def get_latest_tick(self) -> MarketTick:
        """Get latest market tick with real-time updates"""
        try:
            # Use StockFetcher for real-time data
            data = self.fetcher.get_real_time_price(self.symbol)
            
            # Map dictionary to MarketTick
            # Handle timestamp string or datetime object
            ts = data['timestamp']
            if isinstance(ts, str):
                timestamp = datetime.fromisoformat(ts)
            else:
                timestamp = ts

            tick = MarketTick(
                symbol=self.symbol,
                timestamp=timestamp,
                price=float(data['current_price']),
                volume=int(data['volume']),
                bid=None,
                ask=None
            )

            self._last_tick = tick
            logger.debug(f"Fetched tick: {self.symbol} @ ${tick.price:.2f}")
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
        return self.fetcher.is_market_open(self.symbol)


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

            # Check if enough cash (using percentage-based costs)
            trade_value = order.shares * order.price
            trade_cost = trade_value * self.config.transaction_cost_rate
            slippage_cost = trade_value * self.config.slippage_rate
            total_cost = trade_cost + slippage_cost
            total_required = trade_value + total_cost

            if total_required > portfolio.cash:
                return RiskStatus(
                    approved=False,
                    reason=f"Insufficient cash: Need ${total_required:.2f}, have ${portfolio.cash:.2f}"
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
            # Calculate percentage-based costs (matching backtesting)
            trade_value = order.shares * order.price
            trade_cost = trade_value * self.config.transaction_cost_rate
            slippage_cost = trade_value * self.config.slippage_rate
            total_cost = trade_cost + slippage_cost
            total_required = trade_value + total_cost
            return total_required <= portfolio.cash

        elif order.action == TradingAction.SELL:
            position = portfolio.positions.get(order.symbol)
            return position is not None and position.shares >= order.shares

        return False

    def execute(self, order: Order, portfolio: Portfolio) -> Trade:
        """Execute order and update portfolio (matching backtesting cost calculation)"""

        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:06d}"

        if order.action in (TradingAction.BUY_SMALL, TradingAction.BUY_LARGE):
            # Execute buy - use percentage-based costs matching backtesting
            trade_value = order.shares * order.price
            trade_cost = trade_value * self.config.transaction_cost_rate
            slippage_cost = trade_value * self.config.slippage_rate
            total_cost = trade_cost + slippage_cost

            # Deduct trade value + costs from cash
            total_required = trade_value + total_cost
            portfolio.cash -= total_required

            # Update or create position (costs deducted from cash, not included in avg price)
            if order.symbol in portfolio.positions:
                pos = portfolio.positions[order.symbol]
                total_shares = pos.shares + order.shares
                # Calculate avg entry price WITHOUT transaction costs (costs shown separately in trade P&L)
                total_cost_basis = (pos.avg_entry_price * pos.shares) + trade_value
                pos.avg_entry_price = total_cost_basis / total_shares
                pos.shares = total_shares
                pos.current_price = order.price
            else:
                # For new position, avg entry price is just the purchase price (not including costs)
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
                pnl=-total_cost,  # Show transaction cost as negative P&L
                commission=total_cost,  # Store total transaction costs (fee + slippage)
                trade_id=trade_id
            )

        elif order.action == TradingAction.SELL:
            # Execute sell - use percentage-based costs matching backtesting
            position = portfolio.positions[order.symbol]

            trade_value = order.shares * order.price
            trade_cost = trade_value * self.config.transaction_cost_rate
            slippage_cost = trade_value * self.config.slippage_rate
            total_cost = trade_cost + slippage_cost

            # Calculate proceeds after costs
            total_proceeds = trade_value - total_cost

            # Calculate P&L (profit/loss from price difference minus costs)
            pnl = (order.price - position.avg_entry_price) * order.shares - total_cost

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
                commission=total_cost,  # Store total transaction costs (fee + slippage)
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
        from .model_utils import load_rl_agent, normalize_model_path

        try:
            # Normalize path using centralized function (handles all legacy formats)
            normalized_path = normalize_model_path(agent_path)

            # Use centralized loader that handles PPO, RecurrentPPO, Ensemble
            # Note: We pass None for env here because env is not created yet
            # We will handle VecNormalize wrapping in setup_environment
            self.agent = load_rl_agent(normalized_path, env=None)
            logger.info(f"Successfully loaded agent from {normalized_path}")

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
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

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

            # Infer configuration from feature count
            # Feature breakdown:
            # - Base: 5, Tech: 5, Trend: 3 (optional)
            # - Regime: 7 (optional), MTF: 6 (optional)
            # - ActionMask: 4 (old models) or 6 (new models with improved actions)

            use_regime = training_config.get('use_regime_detector', True)
            use_mtf = training_config.get('use_mtf_features', True)
            use_improved = training_config.get('use_improved_actions', True)

            # Calculate expected features based on config
            base_features = 5 + 5  # base + technical
            if use_regime:
                base_features += 7
            if use_mtf:
                base_features += 6

            # Detect trend indicators from feature count
            trend_features = 0
            if use_regime and use_mtf:
                trend_features = 3
                if expected_features == base_features + 3:
                    include_trend_indicators = True
                    logger.info(f"Detected include_trend_indicators=True from observation shape {obs_shape}")
                elif expected_features == base_features:
                    include_trend_indicators = False
                    logger.info(f"Detected include_trend_indicators=False from observation shape {obs_shape}")
                else:
                    logger.warning(f"Unexpected feature count {expected_features}, using config defaults")
                    include_trend_indicators = training_config.get('include_trend_indicators', False)
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

        # Wrap with VecNormalize if stats exist
        # This is critical if the model was trained with normalized observations
        model_path = Path(self.config.agent_path)
        if model_path.is_file():
            model_dir = model_path.parent
        else:
            model_dir = model_path
            
        vec_path = model_dir / "vec_normalize.pkl"
        if not vec_path.exists():
            vec_path = model_dir.parent / "vec_normalize.pkl"
            
        if vec_path.exists():
            try:
                # Wrap env in DummyVecEnv as required by VecNormalize
                self.env = DummyVecEnv([lambda: self.env])
                self.env = VecNormalize.load(str(vec_path), self.env)
                self.env.training = False  # Critical: Do not update stats during inference
                self.env.norm_reward = False  # Critical: Do not normalize rewards
                logger.info(f"Loaded VecNormalize stats from {vec_path}")
            except Exception as e:
                logger.warning(f"Failed to load VecNormalize stats: {e}")

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
        """Find best performing model (algorithm + path) for a symbol using backtest-based scoring"""
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

                # FIX #2: Use backtest performance for scoring instead of hardcoded preferences
                score = 0.0
                backtest_file = latest / "backtest_results.json"

                if backtest_file.exists():
                    try:
                        with open(backtest_file, 'r') as f:
                            backtest_data = json.load(f)
                            # Primary: Sharpe ratio (up to 100 points, scaled by 20x)
                            sharpe = backtest_data.get('sharpe_ratio', 0)
                            sharpe_score = sharpe * 20  # e.g., 2.5 Sharpe = 50 points

                            # Secondary: Total return percentage (scaled)
                            return_pct = backtest_data.get('total_return_pct', 0)
                            return_score = return_pct * 2  # e.g., 20% return = 40 points

                            score = sharpe_score + return_score
                            logger.debug(f"{agent_type} {symbol}: Sharpe {sharpe:.2f} ({sharpe_score:.0f}pts) + Return {return_pct:.2f}% ({return_score:.0f}pts) = {score:.0f}pts")
                    except Exception as e:
                        logger.debug(f"Could not load backtest for {latest.name}: {e}")
                        score = 0

                # Fallback: If no backtest available, use conservative algorithm preference + recency
                if score == 0:
                    # Conservative fallback scoring (much lower than backtest-based)
                    algo_scores = {
                        'recurrent_ppo': 30,  # Fallback preference
                        'ppo': 25,
                        'ensemble': 20
                    }
                    score = algo_scores.get(agent_type_lower, 0)

                    # Add recency bonus
                    model_age_days = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).days
                    recency_score = max(0, 10 - model_age_days)  # Up to 10 points
                    score += recency_score

                    logger.debug(f"{agent_type} {symbol}: No backtest, using fallback score {score:.0f}pts")

                # Add training quality bonus (up to 20 points)
                training_config_file = latest / "training_config.json"
                if training_config_file.exists():
                    try:
                        with open(training_config_file, 'r') as f:
                            config = json.load(f)
                            timesteps = config.get('total_timesteps', 0)
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
                    'score': score
                })

        if not model_candidates:
            return None

        # Select model with highest score (backtest performance dominates)
        best_candidate = max(model_candidates, key=lambda x: x['score'])

        # Use centralized path normalization
        from .model_utils import normalize_model_path
        try:
            normalized_path = normalize_model_path(
                best_candidate['path'],
                agent_type=best_candidate['agent_type']
            )
            logger.info(f"Selected {best_candidate['agent_type']} for {symbol} (score: {best_candidate['score']:.0f})")
            return (best_candidate['agent_type'], str(normalized_path))
        except FileNotFoundError as e:
            logger.error(f"Failed to normalize model path: {e}")
            return None

    def _select_best_stock(self, idle_threshold_override: Optional[int] = None) -> Optional[Tuple[str, str]]:
        """Select best stock from watchlist based on agent backtest performance (prioritized) or stock price performance (fallback)
        Args:
            idle_threshold_override: Override the default cycle count for idle penalty (default: 20)
        Returns: (symbol, agent_path) or None
        """
        try:
            from ..tools.portfolio_manager import portfolio_manager
            from pathlib import Path
            import json

            # Minimum performance improvement required to justify rotation (2%)
            MIN_PERFORMANCE_IMPROVEMENT_PCT = 2.0

            # Idle detection: If agent hasn't traded for this many cycles, heavily penalize this stock
            # Use override if provided, otherwise default to 20
            threshold_cycles = idle_threshold_override if idle_threshold_override is not None else 20
            IDLE_PENALTY_PCT = -50.0  # Massive penalty to force rotation
            
            # Recency penalty: Prevent ping-ponging back to stocks we just left
            RECENT_ROTATION_PENALTY_PCT = -30.0
            RECENT_ROTATION_WINDOW_MINUTES = 30

            watchlist = portfolio_manager.load_portfolio("default")
            if not watchlist:
                logger.warning("No stocks in watchlist for auto-selection")
                return None
                
            # Clean up old recent_rotations entries
            now = datetime.now()
            if self.session.recent_rotations:
                self.session.recent_rotations = {
                    sym: ts for sym, ts in self.session.recent_rotations.items()
                    if (now - ts).total_seconds() / 60 < RECENT_ROTATION_WINDOW_MINUTES
                }

            # Evaluate all watchlist stocks (including current symbol)
            stock_scores = []
            current_symbol_performance = None

            for symbol in watchlist:
                # Check if model exists
                model_info = self._find_best_model_for_symbol(symbol)
                if not model_info:
                    continue

                agent_type, agent_path = model_info

                # Try to load backtest results (prioritized)
                backtest_performance = None
                backtest_file = Path(agent_path) / "backtest_results.json"

                if backtest_file.exists():
                    try:
                        with open(backtest_file, 'r') as f:
                            backtest_data = json.load(f)
                            # Use total return as performance metric
                            backtest_performance = backtest_data.get('total_return_pct', None)
                            if backtest_performance is not None:
                                logger.debug(f"Loaded backtest performance for {symbol}: {backtest_performance:.2f}%")
                    except Exception as e:
                        logger.debug(f"Could not load backtest results for {symbol}: {e}")

                # Fallback to stock price performance if no backtest
                if backtest_performance is None:
                    backtest_performance = self._evaluate_stock_performance(symbol, days=5)
                    if backtest_performance is None:
                        continue
                    logger.debug(f"Using 5-day price performance for {symbol}: {backtest_performance:.2f}%")
                else:
                    logger.debug(f"Using backtest performance for {symbol}: {backtest_performance:.2f}%")

                final_performance = backtest_performance
                
                # Apply idle penalty if this is the current stock and it's been idle too long
                if symbol == self.config.symbol and self.session.consecutive_idle_cycles >= threshold_cycles:
                    final_performance += IDLE_PENALTY_PCT
                    logger.warning(
                        f"Auto-select: {symbol} has been idle for {self.session.consecutive_idle_cycles} cycles "
                        f"(threshold: {threshold_cycles}), "
                        f"applying {IDLE_PENALTY_PCT}% penalty (original: {backtest_performance:.2f}%, "
                        f"adjusted: {final_performance:.2f}%)"
                    )
                    
                # Apply recency penalty if we recently rotated away from this stock
                if symbol in self.session.recent_rotations:
                    ts_left = self.session.recent_rotations[symbol]
                    mins_ago = (now - ts_left).total_seconds() / 60
                    final_performance += RECENT_ROTATION_PENALTY_PCT
                    logger.warning(
                        f"Auto-select: {symbol} was rotated away from {mins_ago:.1f} mins ago, "
                        f"applying {RECENT_ROTATION_PENALTY_PCT}% penalty (original: {backtest_performance:.2f}%, "
                        f"adjusted: {final_performance:.2f}%)"
                    )

                stock_scores.append({
                    'symbol': symbol,
                    'algorithm': agent_type,
                    'agent_path': agent_path,
                    'performance': final_performance,
                    'using_backtest': backtest_file.exists()
                })

                # Track current symbol's performance (original, not penalized)
                if symbol == self.config.symbol:
                    current_symbol_performance = backtest_performance

            if not stock_scores:
                logger.warning("No suitable stocks found for auto-selection")
                return None

            # Sort by performance (descending)
            stock_scores.sort(key=lambda x: x['performance'], reverse=True)
            best_stock = stock_scores[0]

            # Only rotate if the best stock is significantly better than current stock
            # NOTE: If we applied a penalty to current stock, its score in stock_scores is low, 
            # so best_stock will likely be something else.
            # However, we also compare against `current_symbol_performance` (original) to assume "improvement".
            # If we penalized the current stock, we should probably ignore the "improvement threshold" check
            # or treat the penalized score as the current score to beat.
            
            # If current stock was penalized, we definitely want to rotate if there's a better option.
            # Let's verify if the best stock is effectively better than the PENALIZED current stock.
            
            effective_current_score = current_symbol_performance
            if self.session.consecutive_idle_cycles >= threshold_cycles:
                 effective_current_score += IDLE_PENALTY_PCT

            if effective_current_score is not None:
                performance_diff = best_stock['performance'] - effective_current_score

                if best_stock['symbol'] == self.config.symbol:
                    # Current stock is still the best even with penalty? Unlikely but possible.
                    perf_type = "backtest" if best_stock['using_backtest'] else "5-day"
                    logger.info(f"Auto-select: {self.config.symbol} remains the best with {effective_current_score:.2f}% {perf_type} performance")
                    return None
                elif performance_diff < MIN_PERFORMANCE_IMPROVEMENT_PCT:
                    # Not enough improvement to justify rotation
                    logger.info(f"Auto-select: {best_stock['symbol']} is only {performance_diff:.2f}% better than {self.config.symbol} (threshold: {MIN_PERFORMANCE_IMPROVEMENT_PCT}%), staying with current stock")
                    return None

            perf_type = "backtest" if best_stock['using_backtest'] else "5-day"
            logger.info(f"Auto-select: {best_stock['symbol']} ({best_stock['algorithm']}) with {best_stock['performance']:.2f}% {perf_type} performance")

            return (best_stock['symbol'], best_stock['agent_path'])

        except Exception as e:
            logger.error(f"Error selecting best stock: {e}")
            return None

    def _force_close_position(self, current_price: float):
        """Force close current position before rotation (using percentage-based costs)"""
        try:
            from .improvements import ImprovedTradingAction

            current_position = self.portfolio.positions.get(self.config.symbol)
            if not current_position or current_position.shares == 0:
                return

            shares_to_sell = current_position.shares
            symbol = self.config.symbol

            # Execute sell order (full position) - use percentage-based costs
            trade_value = shares_to_sell * current_price
            trade_cost = trade_value * self.config.transaction_cost_rate
            slippage_cost = trade_value * self.config.slippage_rate
            total_cost = trade_cost + slippage_cost

            # Update portfolio (proceeds after costs)
            total_proceeds = trade_value - total_cost
            self.portfolio.cash += total_proceeds

            # Calculate realized PnL
            cost_basis = current_position.avg_entry_price * shares_to_sell
            realized_pnl = trade_value - cost_basis - total_cost
            current_position.realized_pnl += realized_pnl

            # Record trade (use proper TradingAction enum to avoid serialization issues)
            # Map ImprovedTradingAction.SELL_ALL (5) to the compatible TradingAction
            # Since TradingAction doesn't have SELL_ALL, we'll use the raw int but wrap properly
            from datetime import datetime as dt
            trade = Trade(
                symbol=symbol,
                action=TradingAction.SELL,  # Use SELL from TradingAction enum
                shares=shares_to_sell,
                price=current_price,
                timestamp=dt.now(),
                pnl=realized_pnl,
                commission=total_cost,  # Store total transaction costs (fee + slippage)
                trade_id=f"T{len(self.portfolio.trades) + 1:06d}"
            )
            self.portfolio.trades.append(trade)

            # Remove position
            del self.portfolio.positions[symbol]

            logger.info(f"Force closed {shares_to_sell} shares of {symbol} @ ${current_price:.2f}, PnL: ${realized_pnl:.2f}")
            self._add_event("FORCE_CLOSE", f"Closed {shares_to_sell} shares @ ${current_price:.2f} for rotation (PnL: ${realized_pnl:+.2f})")

        except Exception as e:
            logger.error(f"Failed to force close position: {e}")
            raise

    def _rotate_to_stock(self, symbol: str, agent_path: str):
        """Rotate to a new stock by updating config and reloading agent"""
        try:
            logger.info(f"{self.session.session_id} - Rotating from {self.config.symbol} to {symbol}")

            # Record that we are leaving the current symbol
            self.session.recent_rotations[self.config.symbol] = datetime.now()

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
            self.session.consecutive_idle_cycles = 0  # Reset idle counter on rotation

            # Add event with model name (don't use _add_event here as we want the NEW symbol shown)
            self.session.add_event("STOCK_ROTATION", f"Rotated to {symbol} ({model_name})")

            logger.info(f"{self.session.session_id} - Successfully rotated to {symbol} with {model_name}")

        except Exception as e:
            logger.error(f"Failed to rotate to {symbol}: {e}")
            self._add_event("ROTATION_FAILED", f"Failed to rotate to {symbol}: {str(e)}")

    def _check_live_signal(self, symbol: str, agent_path: str) -> bool:
        """
        Shadow Mode: specific check to see if an agent generates a BUY signal RIGHT NOW.
        This is expensive (loads model + builds env), so only use for top candidates.
        """
        try:
            from .model_utils import load_rl_agent, load_env_config_from_model
            from .env_factory import EnvConfig, create_enhanced_env
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

            logger.info(f"Shadow Mode: Checking live signal for {symbol}...")

            # 1. Load Agent
            # We use a temporary variable to not disturb the main self.agent
            shadow_agent = load_rl_agent(agent_path, env=None)

            # 2. Setup Temporary Environment (lite version)
            # We need to match the feature expectations of this specific model
            try:
                training_config = load_env_config_from_model(Path(agent_path))
            except:
                training_config = {}

            # Detect features (similar logic to setup_environment)
            include_trend = training_config.get('include_trend_indicators', False)
            use_regime = training_config.get('use_regime_detector', True)
            use_mtf = training_config.get('use_mtf_features', True)
            
            # Quick observation builder for this symbol
            # We need to fetch data for THIS symbol, not self.config.symbol
            # We can reuse _build_observation logic but need to be careful with config/env context
            
            # Create a temporary config object for data fetching context
            temp_config = LiveTradingConfig(symbol=symbol, agent_path=agent_path)
            temp_market_stream = MarketDataStream(temp_config)
            tick = temp_market_stream.get_latest_tick()
            
            # Build observation manually (extracting logic from _build_observation to be independent)
            # For simplicity/robustness, we'll create a temporary Env to handle normalization/features correctly
            # This is slower but guarantees the dimensions match the agent
            
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d") # ample history
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            env_config = EnvConfig(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_balance=100000,
                # Key feature flags
                include_technical_indicators=True,
                include_trend_indicators=include_trend,
                use_regime_detector=use_regime,
                use_mtf_features=use_mtf,
                # Disable extras for speed
                enable_diagnostics=False
            )
            
            shadow_env = create_enhanced_env(env_config)
            
            # Handle Normalization
            model_path = Path(agent_path)
            model_dir = model_path.parent if model_path.is_file() else model_path
            vec_path = model_dir / "vec_normalize.pkl"
            if not vec_path.exists(): 
                vec_path = model_dir.parent / "vec_normalize.pkl"
                
            if vec_path.exists():
                shadow_env = DummyVecEnv([lambda: shadow_env])
                shadow_env = VecNormalize.load(str(vec_path), shadow_env)
                shadow_env.training = False
                shadow_env.norm_reward = False

            # Get latest observation from env
            # We need to reset the env to get the latest state, but standard reset() goes to start_date.
            # In live trading, we need the "now" state.
            # The cleanest way without rewriting the env is to rely on our _build_observation
            # but we need to temporarily mock 'self.config' and 'self.env' 
            
            # SAVE CURRENT STATE
            original_config = self.config
            original_env = self.env
            
            # SWAP STATE
            self.config = temp_config
            self.env = shadow_env
            
            try:
                # Build observation using the main engine's method (now pointing to shadow config)
                obs = self._build_observation(tick)
                
                # Normalize if needed (duplicate logic from trading_cycle because we are bypassing it)
                if isinstance(shadow_env, VecNormalize):
                    obs_batch = obs.reshape(1, *obs.shape)
                    norm_obs_batch = shadow_env.normalize_obs(obs_batch)
                    obs = norm_obs_batch[0]
                
                # Predict
                action, _ = shadow_agent.predict(obs, deterministic=True)
                
                # Check for BUY
                # Actions: 1=BUY_SMALL, 2=BUY_MEDIUM, 3=BUY_LARGE
                is_buy = int(action) in [1, 2, 3]
                
                conf_msg = "BUY" if is_buy else "NO_BUY"
                logger.info(f"Shadow Mode {symbol}: Agent predicted {conf_msg} (Action {action})")
                
                return is_buy
                
            finally:
                # RESTORE STATE
                self.config = original_config
                self.env = original_env
                
        except Exception as e:
            logger.error(f"Shadow Mode failed for {symbol}: {e}")
            return False

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
            
            # Apply VecNormalize if env is wrapped
            # VecNormalize expects (N, features) and returns (N, features)
            from stable_baselines3.common.vec_env import VecNormalize
            if isinstance(self.env, VecNormalize):
                # Ensure batch dim
                obs_batch = observation.reshape(1, *observation.shape)
                norm_obs_batch = self.env.normalize_obs(obs_batch)
                observation = norm_obs_batch[0] # Remove batch dim

            # Initialize environment if not already done
            if self.env is None:
                self.setup_environment()

            # Use ImprovedTradingAction enum (6 actions)
            from .improvements import ImprovedTradingAction

            # Get current position for action masking
            current_position = self.portfolio.positions.get(self.config.symbol)
            current_shares = current_position.shares if current_position else 0

            # FIX #3: Auto-select stock rotation - check even when holding positions
            # Only rotate after giving current stock enough time (minimum 10 cycles or 10 minutes)
            # OR if agent is idle with cash (bypass cooldown)
            MIN_CYCLES_BEFORE_ROTATION = 10
            MIN_MINUTES_BEFORE_ROTATION = 10

            if self.config.auto_select_stock:
                # Increment cycle counter
                self.session.cycles_since_rotation += 1

                # Check if enough time has passed
                cycles_elapsed = self.session.cycles_since_rotation
                time_elapsed_minutes = 0
                if self.session.last_rotation_time:
                    time_elapsed_minutes = (datetime.now() - self.session.last_rotation_time).total_seconds() / 60

                # Check for Idle Cash Condition (bypass cooldown)
                cash_pct = self.portfolio.cash / self.portfolio.total_value
                is_idle_cash = (cash_pct > 0.5) and (self.session.consecutive_idle_cycles >= 3)

                can_rotate = (cycles_elapsed >= MIN_CYCLES_BEFORE_ROTATION or
                             time_elapsed_minutes >= MIN_MINUTES_BEFORE_ROTATION or
                             is_idle_cash)

                if can_rotate:
                    trigger_reason = "Idle Cash" if is_idle_cash else "Cooldown Expired"
                    logger.info(f"{self.session.session_id} - Auto-select mode: Checking for better stock ({trigger_reason})...")
                    
                    # --- NEW SIGNAL-BASED ROTATION LOGIC (SHADOW MODE) ---
                    # If we are sitting on cash and idle, check for ACTIVE SIGNALS elsewhere
                    best_stock = None
                    
                    if is_idle_cash:
                         logger.info(f"Active Signal Scan: High cash ({cash_pct:.1%}) and idle ({self.session.consecutive_idle_cycles} cycles). Checking top candidates for BUY signals...")
                         
                         # Get candidates (force list return logic inside _select_best_stock if needed, 
                         # but for now we'll just modify how we use it)
                         # We need a way to get candidates without picking just one. 
                         # We will assume _select_best_stock returns the best *static* one.
                         # Better approach: Let's iterate the watchlist manually here or modify _select_best_stock.
                         # To avoid breaking changes, we will use _select_best_stock to find the "Static Best".
                         # Then we will also pick 2 random others from watchlist to check for "Surprise Alpha"
                         
                         static_best = self._select_best_stock()
                         
                         target_found = False
                         if static_best:
                             s_sym, s_path = static_best
                             if s_sym != self.config.symbol:
                                 # Check if this "Static Best" actually wants to buy
                                 if self._check_live_signal(s_sym, s_path):
                                     logger.info(f"Signal Found! {s_sym} has active BUY signal.")
                                     best_stock = static_best
                                     target_found = True
                                 else:
                                     logger.info(f"{s_sym} is best on paper, but has NO signal. Continuing scan...")
                         
                         # If static best didn't have a signal, maybe check one more high-potential stock?
                         # (Omitted for speed, but this is where we'd loop)
                         
                         if not target_found:
                             # Fallback to standard rotation if no signal found but penalty is high
                             best_stock = static_best

                    else:
                        # Standard check (performance based)
                        best_stock = self._select_best_stock()

                    if best_stock:
                        new_symbol, new_agent_path = best_stock
                        if new_symbol != self.config.symbol:
                            # If holding position, force close it before rotation
                            if current_shares > 0:
                                logger.info(f"{self.session.session_id} - Closing {current_shares} shares of {self.config.symbol} before rotation to {new_symbol}")
                                self._force_close_position(tick.price)

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
                    logger.debug(f"{self.session.session_id} - Auto-select: Only {cycles_elapsed} cycles since rotation, waiting...")

            # Access underlying environment if wrapped (VecNormalize -> DummyVecEnv -> Env)
            actual_env = self.env
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            if isinstance(actual_env, VecNormalize):
                actual_env = actual_env.venv
            if isinstance(actual_env, DummyVecEnv) or hasattr(actual_env, 'envs'):
                actual_env = actual_env.envs[0]

            # Get action mask from environment
            if hasattr(actual_env, 'action_masker'):
                action_mask = actual_env.action_masker.get_action_mask(
                    cash=self.portfolio.cash,
                    position=current_shares,
                    current_price=tick.price,
                    portfolio_value=self.portfolio.total_value,
                    max_position_pct=self.config.max_position_size
                )
            else:
                # Fallback to default calculation
                action_mask = np.ones(6 if getattr(actual_env, 'use_improved_actions', True) else 4)

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
                
                # Access adaptive sizer from underlying env
                adaptive_sizer = getattr(actual_env, 'adaptive_sizer', None)

                if int(action) in [ImprovedTradingAction.BUY_SMALL, ImprovedTradingAction.BUY_MEDIUM, ImprovedTradingAction.BUY_LARGE]:
                    # Use environment's adaptive sizer
                    if adaptive_sizer:
                        shares = adaptive_sizer.get_buy_size(
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
                    if adaptive_sizer:
                        sell_shares = adaptive_sizer.get_sell_size(
                            action=int(action),
                            position=current_shares,
                            use_improved_actions=True
                        )
                        shares = -sell_shares
                    else:
                        shares = -max(1, current_shares // 2)

                elif int(action) == ImprovedTradingAction.SELL_ALL:
                    # Sell all (use environment's logic)
                    if adaptive_sizer:
                        sell_shares = adaptive_sizer.get_sell_size(
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
                    timestamp=datetime.now()  # Use current time for accurate seconds precision
                )

                # Validate order
                order_risk_status = self.risk_manager.validate_order(order, self.portfolio)

                if order_risk_status.approved and self.order_executor.validate(order, self.portfolio):
                    trade = self.order_executor.execute(order, self.portfolio)
                    self._add_event("TRADE", f"{trade.action.name} {trade.shares} @ ${trade.price:.2f} (cost: ${trade.commission:.2f})")
                    logger.info(f"{self.session.session_id} - ✅ Trade executed: {trade.action.name} {trade.shares} shares @ ${trade.price:.2f}, Cost: ${trade.commission:.2f}, P&L: ${trade.pnl:+.2f}")

                    # Reset idle counter on successful trade
                    self.session.consecutive_idle_cycles = 0

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
            self.session.consecutive_idle_cycles += 1
            logger.info(f"{self.session.session_id} - ⏸️  HOLD - No trade executed (idle cycles: {self.session.consecutive_idle_cycles})")
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
                
            # Access underlying env configuration if wrapped
            actual_env = self.env
            from stable_baselines3.common.vec_env import VecNormalize
            if isinstance(actual_env, VecNormalize):
                actual_env = actual_env.venv
            if hasattr(actual_env, 'envs'):
                actual_env = actual_env.envs[0]

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
            if actual_env.include_technical_indicators:
                expected_features += 5  # technical indicators
            if actual_env.include_trend_indicators:
                expected_features += 3  # trend indicators
            if getattr(actual_env, 'use_regime_detector', False):
                expected_features += 7  # regime features
            if getattr(actual_env, 'use_mtf_features', False):
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
            if actual_env.include_trend_indicators:
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
            if actual_env.include_trend_indicators:
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
            if actual_env.include_trend_indicators:
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

            if actual_env.include_trend_indicators:
                sma_trend_norm = np.clip(sma_trend_last_60, -0.1, 0.1) * 10
                ema_crossover_norm = np.clip(ema_crossover_last_60, -0.2, 0.2) * 5
                price_momentum_norm = np.clip(price_momentum_last_60, -0.1, 0.1) * 10
                features.extend([sma_trend_norm, ema_crossover_norm, price_momentum_norm])

            # --- Regime Features ---
            if getattr(actual_env, 'use_regime_detector', False):
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
            if getattr(actual_env, 'use_mtf_features', False):
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
            
            # Access underlying env configuration if wrapped
            actual_env = self.env
            if self.env:
                from stable_baselines3.common.vec_env import VecNormalize
                if isinstance(self.env, VecNormalize):
                    actual_env = self.env.venv
                if hasattr(actual_env, 'envs'):
                    actual_env = actual_env.envs[0]
            
            if actual_env:
                if actual_env.include_technical_indicators:
                    expected_features += 5
                if actual_env.include_trend_indicators:
                    expected_features += 3
                if getattr(actual_env, 'use_regime_detector', False):
                    expected_features += 7
                if getattr(actual_env, 'use_mtf_features', False):
                    expected_features += 6
            return np.zeros((60, expected_features), dtype=np.float32)
