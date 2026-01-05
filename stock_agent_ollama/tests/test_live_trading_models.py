"""
Tests for Live Trading data models (Portfolio, Position, Trade, etc.)
"""
import pytest
from datetime import datetime
from src.rl.live_trading import (
    Portfolio, Position, Trade, Order, MarketTick,
    TradingStatus, OrderStatus, TradingAction
)


def test_position_creation():
    """Test Position dataclass creation."""
    pos = Position(
        symbol="AAPL",
        shares=100,
        avg_entry_price=150.0,
        current_price=155.0
    )

    assert pos.symbol == "AAPL"
    assert pos.shares == 100
    assert pos.avg_entry_price == 150.0
    assert pos.current_price == 155.0
    assert pos.unrealized_pnl == 0.0
    assert pos.realized_pnl == 0.0


def test_position_update_price():
    """Test Position price update and unrealized P&L calculation."""
    pos = Position(
        symbol="AAPL",
        shares=100,
        avg_entry_price=150.0,
        current_price=150.0
    )

    # Price goes up
    pos.update_price(160.0)
    assert pos.current_price == 160.0
    assert pos.unrealized_pnl == 1000.0  # (160 - 150) * 100

    # Price goes down
    pos.update_price(145.0)
    assert pos.current_price == 145.0
    assert pos.unrealized_pnl == -500.0  # (145 - 150) * 100


def test_position_serialization():
    """Test Position to_dict and from_dict."""
    pos = Position(
        symbol="TSLA",
        shares=50,
        avg_entry_price=200.0,
        current_price=210.0,
        unrealized_pnl=500.0,
        realized_pnl=100.0
    )

    # Serialize
    data = pos.to_dict()
    assert data['symbol'] == "TSLA"
    assert data['shares'] == 50
    assert data['avg_entry_price'] == 200.0

    # Deserialize
    pos2 = Position.from_dict(data)
    assert pos2.symbol == pos.symbol
    assert pos2.shares == pos.shares
    assert pos2.avg_entry_price == pos.avg_entry_price
    assert pos2.unrealized_pnl == pos.unrealized_pnl


def test_portfolio_creation():
    """Test Portfolio creation."""
    portfolio = Portfolio(
        initial_cash=100000.0,
        cash=100000.0
    )

    assert portfolio.initial_cash == 100000.0
    assert portfolio.cash == 100000.0
    assert len(portfolio.positions) == 0
    assert len(portfolio.trades) == 0


def test_portfolio_total_value():
    """Test Portfolio total_value property."""
    portfolio = Portfolio(
        initial_cash=100000.0,
        cash=50000.0
    )

    # Add positions
    portfolio.positions['AAPL'] = Position(
        symbol='AAPL',
        shares=100,
        avg_entry_price=150.0,
        current_price=160.0
    )
    portfolio.positions['TSLA'] = Position(
        symbol='TSLA',
        shares=50,
        avg_entry_price=200.0,
        current_price=220.0
    )

    # Total value = cash + position values
    # = 50000 + (100 * 160) + (50 * 220)
    # = 50000 + 16000 + 11000 = 77000
    assert portfolio.total_value == 77000.0


def test_portfolio_pnl_properties():
    """Test Portfolio P&L calculation properties."""
    portfolio = Portfolio(
        initial_cash=100000.0,
        cash=90000.0
    )

    portfolio.positions['AAPL'] = Position(
        symbol='AAPL',
        shares=100,
        avg_entry_price=100.0,
        current_price=120.0  # +$20 per share
    )

    # Total value = 90000 + (100 * 120) = 102000
    # Total P&L = 102000 - 100000 = 2000
    # P&L % = (2000 / 100000) * 100 = 2%

    assert portfolio.total_value == 102000.0
    assert portfolio.total_pnl == 2000.0
    assert portfolio.total_pnl_pct == 2.0


def test_portfolio_update_valuations():
    """Test Portfolio update_valuations method."""
    portfolio = Portfolio(
        initial_cash=100000.0,
        cash=85000.0
    )

    portfolio.positions['AAPL'] = Position(
        symbol='AAPL',
        shares=100,
        avg_entry_price=150.0,
        current_price=150.0
    )

    # Create tick with new price
    tick = MarketTick(
        symbol='AAPL',
        timestamp=datetime.now(),
        price=160.0,
        volume=1000000
    )

    portfolio.update_valuations(tick)

    assert portfolio.positions['AAPL'].current_price == 160.0
    assert portfolio.positions['AAPL'].unrealized_pnl == 1000.0


def test_portfolio_serialization():
    """Test Portfolio to_dict and from_dict."""
    portfolio = Portfolio(
        initial_cash=100000.0,
        cash=80000.0
    )

    portfolio.positions['AAPL'] = Position(
        symbol='AAPL',
        shares=100,
        avg_entry_price=150.0,
        current_price=160.0
    )

    # Serialize
    data = portfolio.to_dict()
    assert data['initial_cash'] == 100000.0
    assert data['cash'] == 80000.0
    assert 'AAPL' in data['positions']

    # Deserialize
    portfolio2 = Portfolio.from_dict(data)
    assert portfolio2.initial_cash == portfolio.initial_cash
    assert portfolio2.cash == portfolio.cash
    assert 'AAPL' in portfolio2.positions
    assert portfolio2.positions['AAPL'].shares == 100


def test_trade_creation():
    """Test Trade dataclass creation."""
    trade = Trade(
        symbol="AAPL",
        action=TradingAction.BUY_SMALL,
        shares=100,
        price=150.0,
        timestamp=datetime.now(),
        pnl=-75.0,
        commission=75.0,
        trade_id="T000001"
    )

    assert trade.symbol == "AAPL"
    assert trade.shares == 100
    assert trade.price == 150.0
    assert trade.pnl == -75.0
    assert trade.commission == 75.0


def test_trade_serialization():
    """Test Trade to_dict and from_dict."""
    now = datetime.now()
    trade = Trade(
        symbol="TSLA",
        action=TradingAction.SELL,
        shares=50,
        price=200.0,
        timestamp=now,
        pnl=500.0,
        commission=50.0,
        trade_id="T000002"
    )

    # Serialize
    data = trade.to_dict()
    assert data['symbol'] == "TSLA"
    assert data['shares'] == 50
    assert isinstance(data['timestamp'], str)  # ISO format

    # Deserialize
    trade2 = Trade.from_dict(data)
    assert trade2.symbol == trade.symbol
    assert trade2.shares == trade.shares
    assert trade2.pnl == trade.pnl


def test_order_creation():
    """Test Order dataclass creation."""
    order = Order(
        symbol="AAPL",
        action=TradingAction.BUY_SMALL,
        shares=100,
        price=150.0,
        timestamp=datetime.now()
    )

    assert order.symbol == "AAPL"
    assert order.action == TradingAction.BUY_SMALL
    assert order.shares == 100
    assert order.price == 150.0
    assert order.status == OrderStatus.PENDING


def test_market_tick_creation():
    """Test MarketTick dataclass creation."""
    tick = MarketTick(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=155.50,
        volume=1000000,
        bid=155.45,
        ask=155.55
    )

    assert tick.symbol == "AAPL"
    assert tick.price == 155.50
    assert tick.volume == 1000000
    assert tick.bid == 155.45
    assert tick.ask == 155.55


def test_trading_status_enum():
    """Test TradingStatus enum values."""
    assert TradingStatus.IDLE.value == "idle"
    assert TradingStatus.RUNNING.value == "running"
    assert TradingStatus.PAUSED.value == "paused"
    assert TradingStatus.STOPPED.value == "stopped"
    assert TradingStatus.HALTED.value == "halted"


def test_order_status_enum():
    """Test OrderStatus enum values."""
    assert OrderStatus.PENDING.value == "pending"
    assert OrderStatus.EXECUTED.value == "executed"
    assert OrderStatus.REJECTED.value == "rejected"
