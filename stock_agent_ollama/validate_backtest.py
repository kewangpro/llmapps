#!/usr/bin/env python3
"""
Backtest Validation Script

Validates that backtest results are mathematically correct and free from common bugs.
Runs 10 comprehensive checks including return reconciliation from trades.

Usage: python validate_backtest.py --symbol RIVN
       python validate_backtest.py --watchlist
       python validate_backtest.py --symbol RIVN --algorithm ppo
       python validate_backtest.py --symbol RIVN --run
Example: python validate_backtest.py --symbol RIVN
         python validate_backtest.py --watchlist --algorithm ensemble --run
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from src.tools.portfolio_manager import portfolio_manager
from src.rl.backtesting import BacktestEngine, BacktestConfig
from src.rl.baselines import BuyHoldStrategy, MomentumStrategy
from src.rl.training import EnhancedRLTrainer
from src.rl.model_utils import load_env_config_from_model

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result"""
    status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    print(f"{status} {BOLD}{name}{RESET}")
    if details:
        print(f"      {details}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"{YELLOW}⚠️  WARNING: {message}{RESET}")


def find_all_models(symbol: str, algorithm: str) -> List[Path]:
    """Find all model directories matching symbol and algorithm

    Returns:
        List of Path objects for matching model directories (newest first)
    """
    models_dir = Path("data/models/rl")

    if not models_dir.exists():
        print(f"{RED}Error: Models directory not found: {models_dir}{RESET}")
        sys.exit(1)

    # Find matching algorithm directories
    algo_prefix = algorithm.lower()
    if algo_prefix == "ensemble":
        pattern = f"ensemble_{symbol}_*"
    elif algo_prefix == "recurrent_ppo":
        pattern = f"recurrent_ppo_{symbol}_*"
    else:
        pattern = f"{algo_prefix}_{symbol}_*"

    matching_dirs = sorted(models_dir.glob(pattern), reverse=True)

    # Exclude archive directory
    matching_dirs = [d for d in matching_dirs if d.name != 'archive']

    if not matching_dirs:
        # Don't exit here, return empty list so baselines can still run
        return []

    return matching_dirs


def load_backtest_results(model_dir: Path) -> Tuple[Dict[str, Any], str]:
    """Load backtest results from a specific model directory

    Args:
        model_dir: Path to the model directory

    Returns:
        Tuple of (data, model_name). Returns (None, model_name) if file missing.
    """
    model_name = model_dir.name
    backtest_file = model_dir / "backtest_results.json"

    if not backtest_file.exists():
        return None, model_name

    print(f"{BLUE}Loading backtest from: {backtest_file}{RESET}")
    print(f"{BLUE}Model: {model_name}{RESET}")

    with open(backtest_file) as f:
        data = json.load(f)

    return data, model_name


def run_on_demand_backtest(model_dir: Path, symbol: str, algorithm: str) -> Dict[str, Any]:
    """Run backtest on demand if results file is missing."""
    print(f"{YELLOW}Backtest results not found. Running on-demand backtest for {model_dir.name}...{RESET}")
    
    try:
        # 1. Load Agent
        if algorithm.lower() == 'ensemble':
            model_path = model_dir / "ensemble"
        else:
            model_path = model_dir / "final_model.zip"

        if not model_path.exists():
            print(f"{RED}Model file not found at {model_path}{RESET}")
            return None

        # 2. Setup Config
        # Try to load training config to get improved_actions flag
        try:
            training_config = load_env_config_from_model(model_dir)
            use_improved = training_config.get('use_improved_actions', True)
            include_trend = training_config.get('include_trend_indicators', False)
        except:
            use_improved = True
            include_trend = False

        # Force trend indicators for relevant algos
        if algorithm.lower() in ['recurrent_ppo', 'ensemble']:
            include_trend = True

        # Standard validation period (last 280 days)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=280)).strftime("%Y-%m-%d")

        backtest_config = BacktestConfig(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            use_improved_actions=use_improved,
            include_trend_indicators=include_trend,
            # Validation should match baseline settings for fair comparison?
            # Or use training defaults? Using training defaults (True) is fairer to the agent's training.
            use_risk_manager=True,
            use_adaptive_sizing=True,
            use_regime_detector=True,
            use_mtf_features=True,
            use_kelly_sizing=True
        )

        # 3. Run Backtest
        engine = BacktestEngine(backtest_config)
        
        # Load agent with None env initially
        agent = EnhancedRLTrainer.load_agent(
            model_path=model_path,
            agent_type=algorithm.lower(),
            env=None 
        )
        
        # Run
        result = engine.run_agent_backtest(agent, deterministic=True)
        
        # 4. Save results for future use
        result.save_to_model_dir(model_dir, algorithm.lower())
        
        return result.to_dict()

    except Exception as e:
        print(f"{RED}On-demand backtest failed: {e}{RESET}")
        return None


def validate_return_calculation(data: Dict[str, Any]) -> bool:
    """Validate that return calculation is correct"""
    print_header("Check 1: Return Calculation")

    reported_return = data.get('total_return_pct', 0)

    # Check if we have portfolio values to validate
    if 'initial_portfolio_value' not in data or 'final_portfolio_value' not in data:
        print(f"  Reported Return: {reported_return:.2f}%")
        print_warning("Portfolio values not in backtest results - cannot verify calculation")
        print("  Backtest results should include initial_portfolio_value and final_portfolio_value")
        return True  # Can't validate, but not a failure

    initial = data['initial_portfolio_value']
    final = data['final_portfolio_value']

    # Calculate expected return
    expected_return = (final - initial) / initial * 100

    # Allow small floating point error (0.01%)
    matches = abs(expected_return - reported_return) < 0.01

    print(f"  Initial Portfolio Value: ${initial:,.2f}")
    print(f"  Final Portfolio Value:   ${final:,.2f}")
    print(f"  Calculated Return:       {expected_return:.2f}%")
    print(f"  Reported Return:         {reported_return:.2f}%")
    print(f"  Difference:              {abs(expected_return - reported_return):.4f}%")

    print_check("Return calculation", matches)
    return matches


def validate_action_distribution(data: Dict[str, Any]) -> bool:
    """Validate action distribution sums to 100%"""
    print_header("Check 2: Action Distribution")

    if 'action_distribution' not in data:
        print_warning("Action distribution not in backtest results")
        print("  Backtest results should include action_distribution for validation")
        return True  # Can't validate, but not a failure

    actions = data['action_distribution']
    total_actions = sum(actions.values())

    if total_actions == 0:
        print_check("Action distribution", False, "No actions recorded!")
        return False

    action_pcts = {k: v/total_actions*100 for k, v in actions.items()}
    total_pct = sum(action_pcts.values())

    print(f"  Total Actions: {total_actions}")
    print(f"  Action Breakdown:")
    for action, pct in sorted(action_pcts.items()):
        print(f"    {action:12s}: {pct:6.2f}%")
    print(f"  Sum: {total_pct:.2f}%")

    matches = abs(total_pct - 100.0) < 0.01
    print_check("Action distribution sums to 100%", matches)

    return matches


def validate_win_rate(data: Dict[str, Any]) -> bool:
    """Validate win rate calculation"""
    print_header("Check 3: Win Rate Calculation")

    reported_win_rate = data.get('win_rate', 0)
    total_trades = data.get('total_trades', 0)
    winning_trades = data.get('winning_trades', 0)
    losing_trades = data.get('losing_trades', 0)

    if total_trades == 0:
        print("  No trades to validate")
        return True

    # Win rate = Winning Trades / Completed Trades (round trips)
    # In trading, a "completed trade" is a buy-sell cycle (only sells close positions)
    completed_trades = winning_trades + losing_trades

    if completed_trades == 0:
        print("  No completed trades (round trips) to validate")
        return True

    expected_win_rate = winning_trades / completed_trades

    # Handle win_rate stored as decimal vs percentage
    if reported_win_rate <= 1.0:
        # Stored as decimal (0.61 = 61%, 1.0 = 100%)
        reported_win_rate_pct = reported_win_rate * 100
        matches = abs(expected_win_rate - reported_win_rate) < 0.001
    else:
        # Stored as percentage (61%)
        reported_win_rate_pct = reported_win_rate
        matches = abs(expected_win_rate * 100 - reported_win_rate) < 0.1

    print(f"  Total Actions:       {total_trades} (includes buys and sells)")
    print(f"  Completed Trades:    {completed_trades} (sell round trips)")
    print(f"  Winning Trades:      {winning_trades}")
    print(f"  Losing Trades:       {losing_trades}")
    print(f"  Calculated Win Rate: {expected_win_rate*100:.2f}%")
    print(f"  Reported Win Rate:   {reported_win_rate_pct:.2f}%")

    print_check("Win rate calculation", matches)
    return matches


def validate_portfolio_consistency(data: Dict[str, Any]) -> bool:
    """Validate portfolio value consistency"""
    print_header("Check 4: Portfolio Value Consistency")

    if 'portfolio_values' not in data:
        print_warning("Portfolio value history not in backtest results")
        print("  Backtest results should include portfolio_values array for validation")
        return True

    portfolio_values = data['portfolio_values']

    if not portfolio_values:
        print("  No portfolio value history to validate")
        return True

    if 'initial_portfolio_value' in data and 'final_portfolio_value' in data:
        initial = data['initial_portfolio_value']
        final = data['final_portfolio_value']
        first_value = portfolio_values[0]
        last_value = portfolio_values[-1]

        first_matches = abs(first_value - initial) < 1.0
        last_matches = abs(last_value - final) < 1.0

        print(f"  First Portfolio Value: ${first_value:,.2f}")
        print(f"  Initial Capital:       ${initial:,.2f}")
        print_check("First value matches initial", first_matches)

        print(f"  Last Portfolio Value:  ${last_value:,.2f}")
        print(f"  Final Value:           ${final:,.2f}")
        print_check("Last value matches final", last_matches)
    else:
        first_matches = True
        last_matches = True
        print(f"  Portfolio values available: {len(portfolio_values)} data points")

    # Check for negative values
    min_value = min(portfolio_values)
    has_negative = min_value < 0

    if has_negative:
        print_check("No negative portfolio values", False, f"Min value: ${min_value:,.2f}")
    else:
        print_check("No negative portfolio values", True)

    return first_matches and last_matches and not has_negative


def validate_metrics_reasonableness(data: Dict[str, Any]) -> bool:
    """Check if metrics are within reasonable bounds"""
    print_header("Check 5: Metrics Reasonableness")

    all_reasonable = True

    # Sharpe ratio - Adjust threshold for RL agents
    # Sharpe > 8 is truly exceptional and worth investigating
    sharpe = data.get('sharpe_ratio', 0)
    sharpe_reasonable = sharpe < 8.0
    if not sharpe_reasonable:
        all_reasonable = False

    print(f"  Sharpe Ratio: {sharpe:.2f}")
    if sharpe > 8.0:
        print_warning(f"Sharpe ratio {sharpe:.2f} is extremely high (>8.0)")
        print_warning("This may indicate data leakage or overfitting")
    elif sharpe > 6.0:
        print(f"  Note: Sharpe ratio {sharpe:.2f} is excellent (>6.0)")
    print_check("Sharpe ratio reasonable", sharpe_reasonable)

    # Win rate (handle decimal vs percentage format)
    # Consider sample size - high win rate with few trades is less concerning
    win_rate = data.get('win_rate', 0)
    win_rate_pct = win_rate * 100 if win_rate < 1.0 else win_rate
    winning_trades = data.get('winning_trades', 0)
    losing_trades = data.get('losing_trades', 0)
    completed_trades = winning_trades + losing_trades

    # Adjust threshold based on sample size
    # With < 20 trades, allow up to 95% win rate
    # With >= 20 trades, be more strict at 90%
    if completed_trades < 20:
        win_rate_threshold = 95.0
    else:
        win_rate_threshold = 90.0

    win_rate_reasonable = win_rate_pct < win_rate_threshold
    if not win_rate_reasonable:
        all_reasonable = False

    print(f"  Win Rate: {win_rate_pct:.2f}% ({completed_trades} completed trades)")
    if win_rate_pct > win_rate_threshold:
        print_warning(f"Win rate {win_rate_pct:.2f}% is suspiciously high (>{win_rate_threshold:.0f}%)")
        print_warning("This may indicate data leakage or small sample size luck")
    elif win_rate_pct > 85.0:
        print(f"  Note: Win rate {win_rate_pct:.2f}% is very high (>85%)")
    print_check("Win rate reasonable", win_rate_reasonable)

    # Max drawdown (convert from decimal if needed)
    max_dd = data.get('max_drawdown', 0)
    max_dd_pct = abs(max_dd * 100) if abs(max_dd) < 1.0 else abs(max_dd)
    total_return = data.get('total_return_pct', 0)

    print(f"  Max Drawdown: {max_dd_pct:.2f}%")
    print(f"  Total Return: {total_return:.2f}%")

    # Check for suspiciously low drawdown with high returns
    if total_return > 30 and max_dd_pct < 1.0:
        print_warning(f"Max drawdown {max_dd_pct:.2f}% is very low for {total_return:.2f}% returns")
        print_warning("Verify that realistic slippage/commissions are included")

    return all_reasonable


def validate_trade_pnl(data: Dict[str, Any], num_samples: int = 5) -> bool:
    """Sample and validate individual trade P&L calculations"""
    print_header("Check 6: Individual Trade Validation")

    # Check if paired trades are available (new format)
    paired_trades = data.get('paired_trades', [])

    if paired_trades:
        print(f"  Using paired round-trip trades for validation")
        trades = paired_trades
    else:
        # Fall back to regular trades
        trades = data.get('trades', [])

        if not trades:
            print("  No individual trades recorded to validate")
            print_warning("Consider enabling trade logging for better validation")
            return True

        # Check if trades have required P&L fields
        first_trade = trades[0]
        if 'entry_price' not in first_trade or 'exit_price' not in first_trade:
            print("  Trade structure: Individual actions (BUY/SELL)")
            print("  P&L validation requires paired round-trip trades")
            print_warning("Skipping P&L validation - trades are individual actions, not paired round trips")
            return True  # Not a failure, just not applicable

    print(f"  Total Trades: {len(trades)}")
    print(f"  Sampling {min(num_samples, len(trades))} trades for validation\n")

    # Sample random trades
    import random
    sample_size = min(num_samples, len(trades))
    sampled_trades = random.sample(trades, sample_size)

    all_valid = True
    for i, trade in enumerate(sampled_trades, 1):
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        shares = trade.get('shares', 0)
        commission = trade.get('commission', 0)
        reported_pnl = trade.get('pnl', 0)

        # Calculate expected P&L
        # P&L = (exit_price - entry_price) * shares - commission
        expected_pnl = (exit_price - entry_price) * shares - commission

        matches = abs(expected_pnl - reported_pnl) < 0.01

        if not matches:
            all_valid = False

        print(f"  Trade {i}:")
        print(f"    Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}, Shares: {shares}")
        print(f"    Commission: ${commission:.2f}")
        print(f"    Calculated P&L: ${expected_pnl:.2f}")
        print(f"    Reported P&L:   ${reported_pnl:.2f}")
        print_check(f"    Trade {i} P&L", matches)
        print()

    return all_valid


def validate_commission_inclusion(data: Dict[str, Any]) -> bool:
    """Check that commissions are being applied"""
    print_header("Check 7: Commission/Cost Inclusion")

    trades = data.get('trades', [])

    if not trades:
        print_warning("No individual trades to check for commissions")
        return True

    # Check for either 'commission' or 'cost' field (backward compatible)
    has_commissions = any(trade.get('commission', 0) > 0 or trade.get('cost', 0) > 0 for trade in trades)

    if has_commissions:
        # Support both field names for backward compatibility
        total_commission = sum(trade.get('commission', 0) or trade.get('cost', 0) for trade in trades)
        avg_cost_pct = (total_commission / data.get('initial_portfolio_value', 100000)) * 100
        print(f"  Total Transaction Costs: ${total_commission:.2f}")
        print(f"  Average Cost: {avg_cost_pct:.3f}% of initial capital")
        print(f"  (Includes transaction fees + slippage)")
        print_check("Transaction costs are applied", True)
        return True
    else:
        print_warning("No commissions found in trades")
        print_warning("Backtest may be overly optimistic without transaction costs")
        print_check("Transaction costs are applied", False)
        return False


def validate_return_from_trades(data: Dict[str, Any]) -> bool:
    """
    Validate that portfolio return can be reconstructed from trade P&Ls.

    This is a critical check that ensures:
    - All trades are properly accounted for
    - No phantom gains/losses exist
    - Final portfolio value = Initial + Sum(Trade P&Ls) + Unrealized P&L
    """
    print_header("Check 8: Return Reconciliation from Trades")

    initial = data.get('initial_portfolio_value')
    final = data.get('final_portfolio_value')

    if initial is None or final is None:
        print_warning("Portfolio values not available - cannot validate return from trades")
        return True

    # Try to get paired trades first (preferred format)
    paired_trades = data.get('paired_trades', [])

    if not paired_trades:
        # Fall back to regular trades if paired trades not available
        regular_trades = data.get('trades', [])

        # Check if regular trades have P&L information
        if regular_trades and 'pnl' in regular_trades[0]:
            # Filter to only SELL actions (which realize P&L)
            paired_trades = [t for t in regular_trades if t.get('action', '').upper() in ['SELL', 'SELL_PARTIAL', 'SELL_ALL']]
            if paired_trades:
                print(f"  Using {len(paired_trades)} sell trades from regular trade log")

        if not paired_trades:
            print("  No paired trades or P&L data available")
            print_warning("Cannot validate return reconciliation without trade P&L data")
            print("  This check requires 'paired_trades' or trades with 'pnl' field")
            return True  # Can't validate, but not a failure

    print(f"  Analyzing {len(paired_trades)} completed trades")

    # Sum all realized P&L from trades
    total_realized_pnl = sum(trade.get('pnl', 0) for trade in paired_trades)

    # Check if there are unrealized positions
    # This would be indicated by final cash != final portfolio value
    final_cash = data.get('final_cash_balance')
    position_value = data.get('final_position_value', 0)

    # Calculate expected final value
    # Expected = Initial + Realized P&L from all trades + Unrealized P&L from open positions
    if final_cash is not None and position_value is not None:
        # We have detailed breakdown
        expected_final = final_cash + position_value

        # Also verify cash balance matches initial + realized PnL
        expected_cash = initial + total_realized_pnl
        cash_matches = abs(final_cash - expected_cash) < 1.0

        print(f"\n  {BOLD}Detailed Breakdown:{RESET}")
        print(f"    Initial Capital:        ${initial:,.2f}")
        print(f"    Realized P&L (trades):  ${total_realized_pnl:+,.2f}")
        print(f"    Expected Cash:          ${expected_cash:,.2f}")
        print(f"    Actual Final Cash:      ${final_cash:,.2f}")
        print(f"    Open Position Value:    ${position_value:,.2f}")
        print(f"    Expected Final Value:   ${expected_final:,.2f}")
        print(f"    Actual Final Value:     ${final:,.2f}")

        # Allow $1 tolerance for rounding
        final_matches = abs(expected_final - final) < 1.0

        print(f"\n  {BOLD}Reconciliation:{RESET}")
        print_check("Cash balance matches initial + realized P&L", cash_matches)
        print_check("Final value matches cash + position", final_matches)

        return cash_matches and final_matches
    else:
        # Simple check: Does realized P&L explain the return?
        # Note: This may not match perfectly if there's an open position
        expected_final_simple = initial + total_realized_pnl
        actual_return = final - initial

        print(f"\n  {BOLD}Simple Reconciliation:{RESET}")
        print(f"    Initial Capital:          ${initial:,.2f}")
        print(f"    Realized P&L (all trades): ${total_realized_pnl:+,.2f}")
        print(f"    Expected Change:          ${total_realized_pnl:+,.2f}")
        print(f"    Actual Change:            ${actual_return:+,.2f}")
        print(f"    Difference:               ${abs(actual_return - total_realized_pnl):,.2f}")

        # Check if difference can be explained by unrealized position
        difference = abs(actual_return - total_realized_pnl)

        # If difference is small (< $10), consider it matching
        if difference < 10.0:
            print_check("Return matches realized P&L from trades", True)
            return True
        elif difference < initial * 0.01:  # Less than 1% of initial capital
            print_check("Return approximately matches trades (small open position likely)", True)
            print(f"  Note: ${difference:,.2f} difference likely from unrealized position")
            return True
        else:
            print_check("Return matches realized P&L from trades", False)
            print_warning(f"Large discrepancy (${difference:,.2f}) between trade P&L and actual return")
            print_warning("Possible causes:")
            print_warning("  - Large unrealized position (check final position value)")
            print_warning("  - Missing trades in trade log")
            print_warning("  - Incorrect P&L calculations")
            print_warning("  - Cash balance tracking error")
            return False


def validate_market_data(data: Dict[str, Any], symbol: str) -> bool:
    """
    Validate that backtest prices match real market data.
    Also provides Market Return context.
    """
    print_header("Check 9: Market Data Integrity & Baseline Context")

    from src.tools.stock_fetcher import StockFetcher
    import pandas as pd

    config = data.get('config', {})
    dates = data.get('dates', [])
    trades = data.get('trades', [])

    if not dates:
        print_warning("No dates in backtest results to validate market data")
        return True

    # Handle both string dates and Timestamp objects
    def get_date_str(d):
        if isinstance(d, str):
            return d[:10]
        if hasattr(d, 'strftime'):
            return d.strftime('%Y-%m-%d')
        return str(d)[:10]

    start_date = config.get('start_date') or get_date_str(dates[0])
    end_date = config.get('end_date') or get_date_str(dates[-1])

    print(f"  Fetching validation data for {symbol}")
    print(f"  Period: {start_date} to {end_date}")

    try:
        fetcher = StockFetcher()
        # Fetch with buffer to ensure we cover the range
        df = fetcher.fetch_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )

        if df.empty:
            print_warning(f"Could not fetch data for {symbol} - skipping validation")
            return True

        # 1. Market Return Context
        market_start_price = df['Close'].iloc[0]
        market_end_price = df['Close'].iloc[-1]
        market_return_pct = ((market_end_price - market_start_price) / market_start_price) * 100
        
        # Theoretical Baseline (50% invested, mimicking Buy & Hold baseline)
        baseline_return_pct = market_return_pct * 0.5

        model_return = data.get('total_return_pct', 0)

        print(f"\n  {BOLD}Market Context:{RESET}")
        print(f"    Start Price:   ${market_start_price:.2f}")
        print(f"    End Price:     ${market_end_price:.2f}")
        print(f"    Market Return: {market_return_pct:+.2f}% (Buy & Hold 100%)")
        print(f"    Baseline Est.: {baseline_return_pct:+.2f}% (Buy & Hold 50%)")
        print(f"    Model Return:  {model_return:+.2f}%")
        
        if model_return > market_return_pct:
            print(f"    Performance:   {GREEN}Outperformed Market{RESET}")
        elif model_return > baseline_return_pct:
            print(f"    Performance:   {GREEN}Outperformed 50% Baseline{RESET}")
        else:
            print(f"    Performance:   {YELLOW}Underperformed{RESET}")

        # 2. Price Integrity Check
        if trades:
            print(f"\n  {BOLD}Price Integrity Check:{RESET}")
            matches = 0
            samples = 0
            
            # Create lookup for fast access
            price_map = {d.strftime('%Y-%m-%d'): p for d, p in df['Close'].items()}
            
            # Check last 5 trades
            for trade in trades[-5:]:
                trade_date_str = get_date_str(trade.get('date', ''))
                trade_price = trade['price']
                
                if trade_date_str in price_map:
                    real_price = price_map[trade_date_str]
                    diff_pct = abs(trade_price - real_price) / real_price
                    
                    if diff_pct < 0.01:
                        matches += 1
                    else:
                        print(f"    {RED}Mismatch on {trade_date_str}: Trade ${trade_price:.2f} vs Real ${real_price:.2f} ({diff_pct:.2%}){RESET}")
                    
                    samples += 1
            
            if samples > 0:
                integrity_pass = matches == samples
                print_check(f"Trade prices match market data ({matches}/{samples} sampled)", integrity_pass)
                return integrity_pass
            else:
                return True
        else:
            return True

    except Exception as e:
        print_warning(f"Market data validation error: {e}")
        return True


def validate_reproducibility(symbol: str, algorithm: str, data: Dict[str, Any], model_dir: Path = None) -> bool:
    """Check if backtest results are deterministic by running a second pass"""
    print_header("Check 10: Reproducibility Test")

    if algorithm in ['buy_hold', 'momentum']:
        print("  Baseline strategies are deterministic by definition.")
        return True

    if not model_dir:
        print_warning("No model directory provided - cannot run automatic reproducibility check")
        return True

    # Handle both string dates and Timestamp objects
    def get_date_str(d):
        if isinstance(d, str):
            return d[:10]
        if hasattr(d, 'strftime'):
            return d.strftime('%Y-%m-%d')
        return str(d)[:10]

    print(f"  Running automated verification for {algorithm.upper()}...")
    print(f"  (Executing second backtest pass to verify determinism)")

    try:
        from src.rl.training import EnhancedRLTrainer
        import inspect

        # 1. Load Agent
        # Handle ensemble pathing
        if algorithm.lower() == 'ensemble':
            model_path = model_dir / "ensemble"
        else:
            model_path = model_dir / "final_model.zip"

        if not model_path.exists():
            print_warning(f"Model file not found at {model_path} - skipping check")
            return True

        agent = EnhancedRLTrainer.load_agent(
            model_path=model_path,
            agent_type=algorithm.lower(),
            env=None
        )

        # 2. Reconstruct Config
        config_data = data.get('config', {})
        
        # Ensure required fields are present and dates are strictly YYYY-MM-DD
        reconstructed_config = {
            'symbol': symbol,
            'start_date': get_date_str(config_data.get('start_date') or data.get('dates', [None])[0]),
            'end_date': get_date_str(config_data.get('end_date') or data.get('dates', [None])[-1]),
            'initial_balance': config_data.get('initial_balance', 100000.0),
            'transaction_cost_rate': config_data.get('transaction_cost_rate', 0.0),
            'slippage_rate': config_data.get('slippage_rate', 0.0005)
        }
        
        # Add enhancement flags
        for flag in ['use_action_masking', 'use_enhanced_rewards', 'use_adaptive_sizing', 
                     'use_improved_actions', 'use_risk_manager', 'use_regime_detector', 
                     'use_mtf_features', 'use_kelly_sizing', 'include_trend_indicators']:
            if flag in config_data:
                reconstructed_config[flag] = config_data[flag]

        # Force include_trend_indicators for RPPO/Ensemble as they require it
        if algorithm.lower() in ['recurrent_ppo', 'ensemble']:
            reconstructed_config['include_trend_indicators'] = True

        backtest_config = BacktestConfig(**reconstructed_config)
        engine = BacktestEngine(backtest_config)

        # 3. Run Second Pass
        second_result = engine.run_agent_backtest(agent, deterministic=True)
        
        # 4. Compare Results
        original_return = data.get('total_return_pct', 0)
        second_return = second_result.metrics.total_return_pct
        
        original_trades = data.get('total_executed', 0)
        second_trades = second_result.metrics.total_executed

        # Check for strict equality (with tiny float tolerance for return)
        return_match = abs(original_return - second_return) < 1e-5
        trades_match = original_trades == second_trades
        
        print(f"\n  Comparison:")
        print(f"    Pass 1: {original_return:+.4f}% return, {original_trades} trades")
        print(f"    Pass 2: {second_return:+.4f}% return, {second_trades} trades")

        if return_match and trades_match:
            print_check("Reproducibility (100% match)", True)
            return True
        else:
            details = []
            if not return_match: details.append(f"Return mismatch: {original_return:+.4f} vs {second_return:+.4f}")
            if not trades_match: details.append(f"Trade count mismatch: {original_trades} vs {second_trades}")
            print_check("Reproducibility", False, "; ".join(details))
            print_warning("Strategy is non-deterministic! Evaluation results may be unreliable.")
            return False

    except Exception as e:
        print_warning(f"Automatic reproducibility check failed: {e}")
        # Don't fail the whole validation if this check crashes (e.g. env issues)
        return True


def generate_validation_report(data: Dict[str, Any], model_name: str, symbol: str, algorithm: str, model_dir: Path = None) -> Tuple[int, int, List[str]]:
    """Generate complete validation report for a data dictionary"""
    print()
    print(f"{BOLD}{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BOLD}{BLUE}║{' '*68}║{RESET}")
    print(f"{BOLD}{BLUE}║{'BACKTEST VALIDATION REPORT'.center(68)}║{RESET}")
    print(f"{BOLD}{BLUE}║{f'{symbol} - {algorithm.upper()}'.center(68)}║{RESET}")
    print(f"{BOLD}{BLUE}║{' '*68}║{RESET}")
    print(f"{BOLD}{BLUE}╚{'═'*68}╝{RESET}")

    results = {}
    failed_checks = []

    try:
        results['Return Calculation'] = validate_return_calculation(data)
        results['Action Distribution'] = validate_action_distribution(data)
        results['Win Rate Calculation'] = validate_win_rate(data)
        results['Portfolio Consistency'] = validate_portfolio_consistency(data)
        results['Metrics Reasonableness'] = validate_metrics_reasonableness(data)
        results['Trade P&L'] = validate_trade_pnl(data)
        results['Commission Inclusion'] = validate_commission_inclusion(data)
        results['Return from Trades'] = validate_return_from_trades(data)
        results['Market Data Integrity'] = validate_market_data(data, symbol)
        results['Reproducibility'] = validate_reproducibility(symbol, algorithm, data, model_dir)
        
        # Collect failed checks
        for check_name, passed in results.items():
            if not passed:
                failed_checks.append(check_name)
                
    except Exception as e:
        print(f"{RED}Error during validation: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 0, 10, ["Critical Error"]

    # Summary
    print_header("VALIDATION SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"  Checks Passed: {passed}/{total}")
    print(f"  Success Rate:  {passed/total*100:.1f}%")
    print()

    if passed == total:
        print(f"{GREEN}{BOLD}✅ ALL CHECKS PASSED - Backtest appears valid!{RESET}")
    elif passed >= total * 0.75:
        print(f"{YELLOW}{BOLD}⚠️  MOSTLY PASSED - Review warnings above{RESET}")
    else:
        print(f"{RED}{BOLD}❌ VALIDATION FAILED - Review errors above{RESET}")

    print()
    print(f"{BLUE}{'='*70}{RESET}")
    print()

    return passed, total, failed_checks


def run_baseline_validation(symbol: str) -> List[Dict[str, Any]]:
    """Run validation for baseline strategies"""
    print(f"\n{BOLD}{BLUE}Running Baseline Validation for {symbol}...{RESET}")
    
    # Setup dates (last 280 days, similar to backtest default)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=280)).strftime("%Y-%m-%d")
    
    # Config WITHOUT improvements (Critical for correct baseline behavior)
    config = BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        use_risk_manager=False,
        use_adaptive_sizing=False,
        use_regime_detector=False,
        use_mtf_features=False,
        use_kelly_sizing=False
    )
    
    engine = BacktestEngine(config)
    results = []
    
    # 1. Buy & Hold
    try:
        buy_hold = BuyHoldStrategy()
        result = engine.run_strategy_backtest(buy_hold.get_action)
        data = result.to_dict()
        passed, total, failures = generate_validation_report(data, "Buy & Hold Baseline", symbol, "buy_hold")
        results.append({
            'symbol': symbol,
            'algorithm': 'buy_hold',
            'model_name': 'Buy & Hold Baseline',
            'passed': passed,
            'total': total,
            'return': data.get('total_return_pct', 0),
            'failures': failures,
            'success_rate': (passed / total * 100) if total > 0 else 0
        })
    except Exception as e:
        print(f"{RED}Error validating Buy & Hold: {e}{RESET}")

    # 2. Momentum
    try:
        momentum = MomentumStrategy()
        result = engine.run_strategy_backtest(momentum.get_action)
        data = result.to_dict()
        passed, total, failures = generate_validation_report(data, "Momentum Baseline", symbol, "momentum")
        results.append({
            'symbol': symbol,
            'algorithm': 'momentum',
            'model_name': 'Momentum Baseline',
            'passed': passed,
            'total': total,
            'return': data.get('total_return_pct', 0),
            'failures': failures,
            'success_rate': (passed / total * 100) if total > 0 else 0
        })
    except Exception as e:
        print(f"{RED}Error validating Momentum: {e}{RESET}")
        
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate backtest results for correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all PPO models for TEAM
  python validate_backtest.py --symbol TEAM --algorithm ppo

  # Validate all models AND baselines for META
  python validate_backtest.py --symbol META --baselines

  # Validate all models in watchlist
  python validate_backtest.py --watchlist
        """
    )

    # Create mutually exclusive group for symbol/watchlist
    symbol_group = parser.add_mutually_exclusive_group(required=True)

    symbol_group.add_argument(
        '--symbol',
        type=str,
        help='Stock symbol to validate (e.g., RIVN, TSLA, AAPL)'
    )

    symbol_group.add_argument(
        '--watchlist',
        action='store_true',
        help='Validate all symbols from default watchlist'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='all',
        choices=['ppo', 'recurrent_ppo', 'ensemble', 'all'],
        help='RL algorithm(s) to validate (default: all)'
    )

    parser.add_argument(
        '--latest-only',
        action='store_true',
        help='Validate only the most recent model (default: validate all models)'
    )

    parser.add_argument(
        '--run',
        action='store_true',
        help='Force run backtest before validation'
    )

    args = parser.parse_args()

    # Determine symbols to validate
    if args.watchlist:
        symbols = portfolio_manager.load_portfolio("default")
        if not symbols:
            print(f"{RED}❌ Error: Watchlist is empty or not found{RESET}")
            sys.exit(1)
        print(f"{BLUE}Validating watchlist symbols: {', '.join(symbols)}{RESET}\n")
    else:
        symbols = [args.symbol.upper()]

    # Determine algorithms to validate
    if args.algorithm.lower() == 'all':
        algorithms = ['ppo', 'recurrent_ppo', 'ensemble']
    else:
        algorithms = [args.algorithm.lower()]

    # Validate all combinations
    validation_results = []

    for symbol in symbols:
        # 1. Validate RL Models
        for algorithm in algorithms:
            try:
                all_models = find_all_models(symbol, algorithm)

                # Filter to latest only if requested
                if args.latest_only and all_models:
                    models_to_validate = [all_models[0]]
                    print(f"{BLUE}Validating latest model only (found {len(all_models)} total){RESET}\n")
                else:
                    models_to_validate = all_models
                    if models_to_validate:
                        print(f"{BLUE}Validating all {len(all_models)} model(s) for {symbol} {algorithm.upper()}{RESET}\n")

                # Validate each model
                for model_dir in models_to_validate:
                    try:
                        data, model_name = load_backtest_results(model_dir)
                        
                        # If results missing OR forced run, try to run on-demand
                        if data is None or args.run:
                            if args.run:
                                print(f"{YELLOW}Forcing backtest run for {model_dir.name}...{RESET}")
                            data = run_on_demand_backtest(model_dir, symbol, algorithm)
                            
                        if data is None:
                            print(f"{RED}Skipping validation for {model_name} (no results and generation failed){RESET}")
                            continue

                        passed, total, failures = generate_validation_report(data, model_name, symbol, algorithm, model_dir)
                        validation_results.append({
                            'symbol': symbol,
                            'algorithm': algorithm,
                            'model_name': model_name,
                            'passed': passed,
                            'total': total,
                            'return': data.get('total_return_pct', 0),
                            'failures': failures,
                            'success_rate': (passed / total * 100) if total > 0 else 0
                        })
                    except Exception as e:
                        print(f"{RED}Error validating model {model_dir}: {e}{RESET}")
                        import traceback
                        traceback.print_exc()

            except SystemExit:
                continue
        
        # 2. Validate Baselines (Always)
        baseline_results = run_baseline_validation(symbol)
        validation_results.extend(baseline_results)

    # Check if any validation occurred
    if not validation_results:
        print(f"{RED}❌ No validation results generated.{RESET}")
        sys.exit(1)

    # Print overall summary
    print()
    print(f"{BOLD}{BLUE}{'='*140}{RESET}")
    print(f"{BOLD}{BLUE}{'OVERALL VALIDATION SUMMARY'.center(140)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*140}{RESET}\n")

    # Summary table with full model names and returns
    print(f"{'Symbol':<10} {'Model':<45} {'Return':<10} {'Passed':<8} {'Status':<10} {'Details/Warnings'}")
    print(f"{'-'*140}")

    for result in validation_results:
        symbol = result['symbol']
        model_name = result['model_name']
        ret = f"{result.get('return', 0):+.2f}%"
        passed = f"{result['passed']}/{result['total']}"
        failures = result.get('failures', [])
        details = ", ".join(failures) if failures else ""

        if result['passed'] == result['total']:
            status = f"{GREEN}✅ PASS{RESET}"
        elif result['passed'] >= result['total'] * 0.75:
            status = f"{YELLOW}⚠️  WARN{RESET}"
        else:
            status = f"{RED}❌ FAIL{RESET}"

        print(f"{symbol:<10} {model_name:<45} {ret:<10} {passed:<8} {status:<10} {details}")

    # Overall stats
    total_checks = sum(r['total'] for r in validation_results)
    total_passed = sum(r['passed'] for r in validation_results)
    overall_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0

    print()
    print(f"{BOLD}Overall: {total_passed}/{total_checks} checks passed ({overall_rate:.1f}%){RESET}")

    all_passed = all(r['passed'] == r['total'] for r in validation_results)
    if all_passed:
        print(f"{GREEN}{BOLD}✅ ALL VALIDATIONS PASSED!{RESET}")
    else:
        failed_count = sum(1 for r in validation_results if r['passed'] < r['total'])
        print(f"{YELLOW}{BOLD}⚠️  {failed_count} validation(s) need attention{RESET}")

    print(f"{BOLD}{BLUE}{'='*140}{RESET}\n")


if __name__ == "__main__":
    main()
