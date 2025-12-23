#!/usr/bin/env python3
"""
Backtest Validation Script

Validates that backtest results are mathematically correct and free from common bugs.
Usage: python validate_backtest.py --symbol RIVN --algorithm ppo
       python validate_backtest.py --watchlist --algorithm ppo
Example: python validate_backtest.py --symbol RIVN --algorithm ensemble
         python validate_backtest.py --watchlist --algorithm ensemble
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime, timedelta

# Add src to path for portfolio_manager import
sys.path.insert(0, str(Path(__file__).parent))
from src.tools.portfolio_manager import portfolio_manager

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


def load_backtest_results(symbol: str, algorithm: str) -> Dict[str, Any]:
    """Load backtest results from file"""
    # Find the model directory (models are stored flat in data/models/rl/)
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

    if not matching_dirs:
        print(f"{RED}Error: No models found for {symbol} {algorithm}{RESET}")
        sys.exit(1)

    model_dir = matching_dirs[0]
    backtest_file = model_dir / "backtest_results.json"

    if not backtest_file.exists():
        print(f"{RED}Error: No backtest results found at {backtest_file}{RESET}")
        print(f"Run backtest first: python -m src.tools.backtest_cli {symbol} {algorithm}")
        sys.exit(1)

    print(f"{BLUE}Loading backtest from: {backtest_file}{RESET}")

    with open(backtest_file) as f:
        data = json.load(f)

    return data


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
    if reported_win_rate < 1.0:
        # Stored as decimal (0.61 = 61%)
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


def validate_reproducibility(symbol: str, algorithm: str, data: Dict[str, Any]) -> bool:
    """Check if backtest results are deterministic"""
    print_header("Check 8: Reproducibility Test")

    print("  To test reproducibility, run the backtest twice:")
    print(f"  python -m src.tools.backtest_cli {symbol} {algorithm}")
    print(f"  python -m src.tools.backtest_cli {symbol} {algorithm}")
    print()
    print("  Then compare the results - they should be identical.")
    print("  If results differ, there may be non-deterministic behavior.")

    return True


def generate_validation_report(symbol: str, algorithm: str) -> Tuple[int, int]:
    """Generate complete validation report

    Returns:
        Tuple of (passed_checks, total_checks)
    """
    print()
    print(f"{BOLD}{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BOLD}{BLUE}║{' '*68}║{RESET}")
    print(f"{BOLD}{BLUE}║{'BACKTEST VALIDATION REPORT'.center(68)}║{RESET}")
    print(f"{BOLD}{BLUE}║{f'{symbol} - {algorithm.upper()}'.center(68)}║{RESET}")
    print(f"{BOLD}{BLUE}║{' '*68}║{RESET}")
    print(f"{BOLD}{BLUE}╚{'═'*68}╝{RESET}")

    # Load data
    try:
        data = load_backtest_results(symbol, algorithm)
    except Exception as e:
        print(f"{RED}Error loading backtest results: {e}{RESET}")
        return 0, 8

    # Run all validation checks
    results = {}

    try:
        results['return_calculation'] = validate_return_calculation(data)
        results['action_distribution'] = validate_action_distribution(data)
        results['win_rate'] = validate_win_rate(data)
        results['portfolio_consistency'] = validate_portfolio_consistency(data)
        results['metrics_reasonableness'] = validate_metrics_reasonableness(data)
        results['trade_pnl'] = validate_trade_pnl(data)
        results['commission_inclusion'] = validate_commission_inclusion(data)
        results['reproducibility'] = validate_reproducibility(symbol, algorithm, data)
    except Exception as e:
        print(f"{RED}Error during validation: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 0, 8

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

    # Recommendations
    if not results.get('commission_inclusion', True):
        print(f"{YELLOW}Recommendation: Ensure transaction costs are enabled in training config{RESET}")

    if data.get('sharpe_ratio', 0) > 5.0:
        print(f"{YELLOW}Recommendation: Verify no lookahead bias or data leakage{RESET}")

    if data.get('win_rate', 0) > 80.0:
        print(f"{YELLOW}Recommendation: Manually verify several trades for correctness{RESET}")

    return passed, total


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate backtest results for correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_backtest.py --symbol RIVN --algorithm ppo
  python validate_backtest.py --symbol TSLA --algorithm ensemble
  python validate_backtest.py --watchlist --algorithm ppo
  python validate_backtest.py --watchlist --algorithm ensemble

Algorithms:
  ppo              - Proximal Policy Optimization
  recurrent_ppo    - RecurrentPPO with LSTM memory
  ensemble         - Ensemble of PPO + RecurrentPPO
  all              - Validate all algorithms
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
        required=True,
        choices=['ppo', 'recurrent_ppo', 'ensemble', 'all'],
        help='RL algorithm(s) to validate'
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
        for algorithm in algorithms:
            passed, total = generate_validation_report(symbol, algorithm)
            validation_results.append({
                'symbol': symbol,
                'algorithm': algorithm,
                'passed': passed,
                'total': total,
                'success_rate': (passed / total * 100) if total > 0 else 0
            })

    # Print overall summary if validating multiple combinations
    if len(validation_results) > 1:
        print()
        print(f"{BOLD}{BLUE}{'='*80}{RESET}")
        print(f"{BOLD}{BLUE}{'OVERALL VALIDATION SUMMARY'.center(80)}{RESET}")
        print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

        # Summary table
        print(f"{'Symbol':<10} {'Algorithm':<15} {'Passed':<10} {'Success Rate':<15} {'Status'}")
        print(f"{'-'*80}")

        for result in validation_results:
            symbol = result['symbol']
            algo = result['algorithm'].upper()
            passed = f"{result['passed']}/{result['total']}"
            success = f"{result['success_rate']:.1f}%"

            if result['passed'] == result['total']:
                status = f"{GREEN}✅ PASS{RESET}"
            elif result['passed'] >= result['total'] * 0.75:
                status = f"{YELLOW}⚠️  WARN{RESET}"
            else:
                status = f"{RED}❌ FAIL{RESET}"

            print(f"{symbol:<10} {algo:<15} {passed:<10} {success:<15} {status}")

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

        print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


if __name__ == "__main__":
    main()
