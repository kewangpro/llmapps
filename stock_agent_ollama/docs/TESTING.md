# 🧪 Testing Guide

Comprehensive testing documentation for the Stock Agent Pro platform.

## Overview

The project includes **135 automated tests** covering critical trading algorithms, data models, and system components. While overall coverage is 19%, **core trading logic is 94-100% covered**.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_technical_analysis_extended.py -v

# Run tests matching pattern
python -m pytest tests/ -k "reward" -v

# Run with detailed output
python -m pytest tests/ -vv --tb=short
```

## Test Organization

### Test Files

```
tests/
├── test_action_masking.py              # 10 tests - Action masking logic
├── test_baseline_strategies.py         # 35 tests - Buy & Hold, Momentum strategies
├── test_config.py                      # 14 tests - Configuration management
├── test_live_trading_models.py         # 14 tests - Portfolio, Position, Trade models
├── test_reward_functions.py            # 29 tests - RL reward calculations
├── test_rl_components.py               # 5 tests - Ensemble, environment factory
├── test_technical_analysis.py          # 5 tests - Basic indicators
└── test_technical_analysis_extended.py # 23 tests - Advanced indicators
```

**Total: 135 tests**

## Coverage by Module

### ✅ Fully Tested (90-100% coverage)

| Module | Coverage | Tests | What's Tested |
|--------|----------|-------|---------------|
| **Baseline Strategies** | 100% | 35 | Buy & Hold, Momentum, SimpleMomentum strategies |
| **Action Masking** | 100% | 10 | Invalid trade prevention, position limits |
| **Live Trading Models** | 100% | 14 | Portfolio, Position, Trade, Order data structures |
| **Technical Analysis** | 99% | 28 | All indicators (RSI, MACD, BB, Stochastic, ATR, trends, signals) |
| **Configuration** | 96% | 14 | Settings, env vars, directory management |
| **Reward Functions** | 94% | 29 | Simple, RiskAdjusted, Customizable rewards |
| **Environment Factory** | 90% | 5 | Environment creation, validation |

### 🟡 Partially Tested (20-50% coverage)

| Module | Coverage | Why Lower |
|--------|----------|-----------|
| RL Environments | 31% | Complex state machines, better tested via integration tests |
| Backtesting | 28% | Large module, integration-heavy |
| Query Processor | 11% | AI/LLM integration, hard to unit test |

### ⚪ Not Tested (0% coverage)

- **UI Components** (src/ui/*) - Tested manually, not via unit tests
- **Session Manager** - File I/O heavy, needs integration tests
- **Model Utils** - TensorFlow model operations, needs end-to-end tests

## Test Categories

### 1. Technical Analysis Tests (28 tests)

**File:** `test_technical_analysis.py`, `test_technical_analysis_extended.py`

**Coverage:** 99% (140/142 statements)

#### What's Tested:
- ✅ Moving Averages (SMA, EMA)
- ✅ Momentum Indicators (RSI, MACD, Stochastic)
- ✅ Volatility Indicators (Bollinger Bands, ATR)
- ✅ Trend Analysis (bullish/bearish detection)
- ✅ Trading Signal Generation (BUY/SELL/HOLD with confidence)
- ✅ Risk Factor Detection
- ✅ Edge Cases (insufficient data, NaN values, zero change)

#### Example:
```python
def test_calculate_bollinger_bands_width(sample_price_data):
    """Test that upper band is above middle, middle above lower."""
    result = TechnicalAnalysis.calculate_bollinger_bands(
        sample_price_data['Close'], window=20, num_std=2
    )

    upper = result['bb_upper'][valid_idx]
    middle = result['bb_middle'][valid_idx]
    lower = result['bb_lower'][valid_idx]

    assert (upper >= middle).all()
    assert (middle >= lower).all()
```

### 2. Reward Function Tests (29 tests)

**File:** `test_reward_functions.py`

**Coverage:** 94% (116/124 statements)

#### What's Tested:
- ✅ **SimpleReturnReward** - Portfolio return-based rewards
- ✅ **RiskAdjustedReward** - Sharpe ratio, volatility, drawdown penalties
- ✅ **CustomizableReward** - Feature toggles (Sharpe, drawdown, costs)
- ✅ Transaction Costs & Slippage
- ✅ Extreme Action Penalties
- ✅ Window Size Management
- ✅ Multi-Episode Support
- ✅ Factory Function

#### Example:
```python
def test_risk_adjusted_reward_drawdown_penalty(default_config):
    """Test drawdown penalty when portfolio value drops from peak."""
    reward_fn = RiskAdjustedReward(default_config)

    reward_fn.calculate(100000.0, 1, 1, 50000.0, 500.0, 100.0, 100.0)
    reward_fn.calculate(120000.0, 1, 1, 50000.0, 500.0, 140.0, 100.0)

    # 15% drawdown
    reward = reward_fn.calculate(102000.0, 1, 1, 50000.0, 500.0, 104.0, 140.0)

    drawdown = (120000.0 - 102000.0) / 120000.0
    assert drawdown == pytest.approx(0.15, rel=0.01)
```

### 3. Baseline Strategy Tests (35 tests)

**File:** `test_baseline_strategies.py`

**Coverage:** 100% (112/112 statements)

#### What's Tested:
- ✅ **BuyHoldStrategy** - Buy once, hold forever
- ✅ **MomentumStrategy** - Lookback-based momentum trading
- ✅ **SimpleMomentumStrategy** - Simple price change momentum
- ✅ Stateless function variants
- ✅ Action space consistency (standard vs improved)
- ✅ Reset functionality
- ✅ Edge cases (NaN, determinism, price extraction)

#### Example:
```python
def test_momentum_strategy_positive_momentum_buy(increasing_prices):
    """Test MomentumStrategy buys on positive momentum."""
    strategy = MomentumStrategy(lookback=10, threshold=0.05)

    for i in range(len(increasing_prices)):
        action = strategy.get_action(
            increasing_prices[i:i+1],
            price=increasing_prices[i, 0]
        )

    # Should buy due to positive momentum
    assert strategy.has_position is True
```

### 4. Action Masking Tests (10 tests)

**File:** `test_action_masking.py`

**Coverage:** 100%

#### What's Tested:
- ✅ Mask generation for invalid actions
- ✅ Insufficient cash detection
- ✅ Position limit enforcement
- ✅ Standard vs Improved action spaces

### 5. Configuration Tests (14 tests)

**File:** `test_config.py`

**Coverage:** 96%

#### What's Tested:
- ✅ Base paths and directory creation
- ✅ Environment variable overrides
- ✅ Cache TTL settings
- ✅ RL trading parameters
- ✅ Transaction cost configuration

### 6. Live Trading Model Tests (14 tests)

**File:** `test_live_trading_models.py`

**Coverage:** 100%

#### What's Tested:
- ✅ Position creation and updates
- ✅ Portfolio valuation and P&L
- ✅ Trade and Order serialization
- ✅ Market tick data structures

### 7. RL Component Tests (5 tests)

**File:** `test_rl_components.py`

**Coverage:** 90%

#### What's Tested:
- ✅ Environment configuration
- ✅ Ensemble voting logic
- ✅ Environment factory

## Running Tests

### Basic Commands

```bash
# All tests with verbose output
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_reward_functions.py -v

# Specific test function
python -m pytest tests/test_reward_functions.py::test_simple_return_reward_positive_return -v

# Tests matching pattern
python -m pytest tests/ -k "momentum" -v
python -m pytest tests/ -k "reward and transaction" -v

# Stop on first failure
python -m pytest tests/ -x

# Show print statements
python -m pytest tests/ -s
```

### Coverage Reports

```bash
# Terminal coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# HTML coverage report (opens in browser)
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Coverage for specific module
python -m pytest tests/ --cov=src/tools/technical_analysis --cov-report=term-missing

# Show only uncovered lines
python -m pytest tests/ --cov=src --cov-report=term-missing | grep -v "100%"
```

### Advanced Options

```bash
# Parallel execution (faster)
python -m pytest tests/ -n auto

# Detailed failure output
python -m pytest tests/ -vv --tb=long

# Show slowest tests
python -m pytest tests/ --durations=10

# Run only failed tests from last run
python -m pytest tests/ --lf

# Warnings as errors (strict mode)
python -m pytest tests/ --strict-warnings
```

## Test Fixtures

Common fixtures used across tests:

### Data Fixtures

```python
@pytest.fixture
def sample_price_data():
    """Generate realistic OHLCV price data."""
    # 100 days of realistic price data with upward trend

@pytest.fixture
def bearish_price_data():
    """Generate bearish (downward trending) price data."""

@pytest.fixture
def sample_observation():
    """Create sample RL environment observation."""

@pytest.fixture
def default_config():
    """Default reward configuration."""
    return RewardConfig()
```

### Using Fixtures

```python
def test_analyze_trends_bullish_scenario(sample_price_data):
    """Test trend analysis on bullish data."""
    result = TechnicalAnalysis.analyze_trends(sample_price_data)
    assert result['overall_trend'] in ['Bullish', 'Neutral']
```

## Writing New Tests

### Best Practices

1. **Test One Thing**
   ```python
   # Good
   def test_buy_hold_buys_on_first_step():
       strategy = BuyHoldStrategy()
       action = strategy.get_action(obs)
       assert action == 3  # BUY_LARGE

   # Bad - tests multiple things
   def test_buy_hold_strategy():
       strategy = BuyHoldStrategy()
       assert strategy.get_action(obs) == 3
       assert strategy.has_bought is True
       assert strategy.get_action(obs) == 0
   ```

2. **Descriptive Names**
   ```python
   # Good
   def test_momentum_strategy_sells_on_negative_momentum():

   # Bad
   def test_momentum():
   ```

3. **Use Fixtures for Setup**
   ```python
   @pytest.fixture
   def initialized_strategy():
       strategy = MomentumStrategy(lookback=10)
       # Setup code
       return strategy

   def test_something(initialized_strategy):
       # Test code
   ```

4. **Test Edge Cases**
   ```python
   def test_calculate_rsi_with_zero_change():
       """Test RSI with constant prices (no change)."""
       constant_data = pd.Series([100.0] * 50)
       rsi = TechnicalAnalysis.calculate_rsi(constant_data)
       assert isinstance(rsi, pd.Series)  # Shouldn't crash
   ```

5. **Assert Error Messages**
   ```python
   def test_invalid_reward_type():
       with pytest.raises(ValueError, match="Unknown reward type"):
           get_reward_function("invalid")
   ```

### Test Template

```python
"""
Tests for [module name].

Brief description of what's being tested.
"""

import pytest
from src.module import ClassOrFunction


@pytest.fixture
def sample_data():
    """Description of fixture."""
    return setup_data()


def test_basic_functionality(sample_data):
    """Test basic functionality works as expected."""
    result = ClassOrFunction.method(sample_data)
    assert result == expected_value


def test_edge_case():
    """Test edge case handling."""
    # Setup edge case
    # Execute
    # Assert correct behavior
```

## Continuous Integration

### GitHub Actions (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Maintenance

### When to Update Tests

- **Code changes:** Update tests when modifying algorithms
- **Bug fixes:** Add test to prevent regression
- **New features:** Write tests before or during implementation
- **Refactoring:** Ensure all tests still pass

### Identifying Flaky Tests

```bash
# Run test 100 times to check for flakiness
python -m pytest tests/test_file.py::test_name --count=100
```

## FAQ

**Q: Why is overall coverage only 19% when critical modules are 94-100%?**

A: The codebase includes large UI modules (src/ui/*), ML model utilities, and integration-heavy components that aren't suited for unit testing. The 19% represents robust coverage of **all critical trading logic**.

**Q: How long do tests take to run?**

A: ~5 seconds for all 135 tests. Individual test files run in <1 second.

**Q: Should I write tests for UI components?**

A: UI components are better tested manually or with end-to-end tests. Focus unit tests on business logic.

**Q: How do I test code that uses external APIs?**

A: Use mocking:
```python
@pytest.fixture
def mock_yfinance(monkeypatch):
    def mock_download(*args, **kwargs):
        return pd.DataFrame(...)
    monkeypatch.setattr("yfinance.download", mock_download)
```

**Q: What's the minimum coverage for a new module?**

A: Aim for 80%+ for business logic modules. Lower coverage is acceptable for integration-heavy or UI code.

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Test Suite Status:** ✅ 135 tests passing | 19% overall coverage | Critical modules 94-100% covered
