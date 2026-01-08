import os
import json
import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

# Constants
MODELS_DIR = Path("data/models/rl")
ARCHIVE_DIR = Path("data/models/archive")  # Archive at models level, not rl level
DEFAULT_WATCHLIST_PATH = Path("data/portfolios/default.json")

# Thresholds for analysis
THRESHOLDS = {
    'good_sharpe': 1.0,
    'poor_sharpe': 0.0,
    'high_drawdown': 0.20,  # 20%
    'overtrading_rate': 2.0,  # Trades per day
    'undertrading_rate': 0.05, # Trades per day
    'action_collapse': 0.80,   # 80% of actions are the same
    'win_rate_concern': 0.40,  # 40%
}

class Color:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def load_json(path: Path) -> Dict[str, Any]:
    """Safely load a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def parse_age_to_seconds(age_str: str) -> int:
    """Parse age string (e.g. '24h', '7d') to seconds."""
    try:
        if not age_str:
            return 0
        unit = age_str[-1].lower()
        if unit.isdigit():
            return int(age_str)
        
        value = int(age_str[:-1])
        if unit == 'h': return value * 3600
        if unit == 'd': return value * 86400
        return value
    except:
        return 0

def get_model_age(path: Path) -> str:
    """Get readable age of a model file."""
    try:
        mtime = path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        delta = datetime.now() - dt
        
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        else:
            return f"{delta.seconds // 60}m ago"
    except:
        return "?"

def analyze_model(model_dir: Path) -> Dict[str, Any]:
    """Analyze a single model directory."""
    backtest_path = model_dir / "backtest_results.json"
    config_path = model_dir / "training_config.json"
    
    if not backtest_path.exists():
        return None
        
    results = load_json(backtest_path)
    config = load_json(config_path)
    
    if not results:
        return None
        
    # Extract baseline results if available (usually at the top level of backtest_results.json for batch runs, 
    # or sometimes embedded. Based on file structure, they might be missing in individual model folders.
    # Let's check if 'baseline_results' key exists or if we need to infer from the text report)
    # Actually, retrain_rl.py often saves baselines in the same JSON or a sibling. 
    # For now, we'll look for 'baseline_results' key in the loaded json.
    
    baselines = results.get('baseline_results', {})
    buy_hold_return = baselines.get('Buy & Hold', {}).get('total_return_pct', 0.0)
    
    # Extract key metrics
    metrics = {
        'name': model_dir.name,
        'symbol': config.get('symbol', results.get('symbol', 'Unknown')),
        'agent_type': config.get('agent_type', 'unknown'),
        'age': get_model_age(backtest_path),
        'mtime': backtest_path.stat().st_mtime,
        'return': results.get('total_return_pct', 0.0),
        'sharpe': results.get('sharpe_ratio', 0.0),
        'drawdown': results.get('max_drawdown', 0.0),
        'win_rate': results.get('win_rate', 0.0),
        'trades': results.get('total_executed', results.get('total_trades', 0)),
        'actions': results.get('action_distribution', {}),
        'initial_balance': results.get('initial_portfolio_value', 0),
        'final_balance': results.get('final_portfolio_value', 0),
        'market_return': 0.0, # Placeholder
        'path': model_dir,
        'buy_hold_return': buy_hold_return
    }
    
    # Calculate derived metrics
    action_counts = metrics['actions']
    total_actions = sum(action_counts.values())
    
    # Check for action collapse
    max_action_pct = 0.0
    dominant_action = "None"
    if total_actions > 0:
        max_count = max(action_counts.values())
        max_action_pct = max_count / total_actions
        dominant_action = max(action_counts, key=action_counts.get)
        
    metrics['max_action_pct'] = max_action_pct
    metrics['dominant_action'] = dominant_action
    
    # Estimate trades per day (approximate if dates missing)
    # Assume 252 trading days per year
    days = 252 * (len(results.get('portfolio_values', [])) / 252)
    if days > 0:
        metrics['trades_per_day'] = metrics['trades'] / days
    else:
        metrics['trades_per_day'] = 0
        
    # Check if market outperformed
    # Note: Requires fetching market data which is slow, using config/results comparison if available
    # For now, we look for 'Buy & Hold' comparison in the same folder structure if comprehensive test was run
    
    return metrics

def prune_models(models: List[Dict[str, Any]], min_return: float, age_seconds: int):
    """Archive models matching criteria."""
    print(f"\n{Color.HEADER}{'='*80}")
    print(f"🧹 PRUNING MODELS (Return < {min_return}%, Age > {age_seconds/3600:.1f}h)")
    print(f"{Color.HEADER}{'='*80}{Color.END}")
    
    threshold_time = datetime.now().timestamp() - age_seconds
    to_prune = []
    
    for m in models:
        # We prune models that are losing money AND older than the threshold
        if m['return'] < min_return and m['mtime'] < threshold_time:
            to_prune.append(m)
            
    if not to_prune:
        print("No models matched the pruning criteria.")
        return
        
    if not ARCHIVE_DIR.exists():
        ARCHIVE_DIR.mkdir(parents=True)
        print(f"Created archive directory: {ARCHIVE_DIR}")
        
    print(f"Found {len(to_prune)} models to archive...\n")
    
    for m in to_prune:
        source_path = m['path']
        dest_path = ARCHIVE_DIR / source_path.name
        
        try:
            # Move directory
            if dest_path.exists():
                # If already exists in archive, remove existing first to allow overwrite/update
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()
                    
            shutil.move(str(source_path), str(dest_path))
            print(f"  {Color.YELLOW}Archived:{Color.END} {m['name']:<45} (Return: {m['return']:+.2f}%, Age: {m['age']})")
        except Exception as e:
            print(f"  {Color.RED}Error archiving {m['name']}:{Color.END} {e}")
            
    print(f"\n{Color.GREEN}Success: Moved {len(to_prune)} models to {ARCHIVE_DIR}{Color.END}")

def generate_insights(models: List[Dict[str, Any]]) -> List[str]:
    """Generate high-level insights based on aggregated model data."""
    insights = []
    
    if not models:
        return ["No models found to analyze."]
        
    avg_return = np.mean([m['return'] for m in models])
    avg_sharpe = np.mean([m['sharpe'] for m in models])
    avg_trades = np.mean([m['trades'] for m in models])
    
    # Insight 1: Overall Performance
    if avg_return < 0:
        insights.append(f"{Color.RED}CRITICAL: Average model return is negative ({avg_return:.2f}%). Strategies are consistently losing money.{Color.END}")
    elif avg_sharpe < 0.5:
        insights.append(f"{Color.YELLOW}WARNING: Risk-adjusted returns are poor (Avg Sharpe: {avg_sharpe:.2f}). Models take too much risk for the reward.{Color.END}")
    else:
        insights.append(f"{Color.GREEN}POSITIVE: Models are showing promise (Avg Return: {avg_return:.2f}%, Avg Sharpe: {avg_sharpe:.2f}).{Color.END}")
        
    # Insight 2: Trading Frequency
    overtrading_models = [m for m in models if m['trades_per_day'] > THRESHOLDS['overtrading_rate']]
    if len(overtrading_models) > len(models) * 0.3:
        insights.append(f"{Color.YELLOW}ISSUE: {len(overtrading_models)} models ({len(overtrading_models)/len(models):.0%}) are over-trading (>2 trades/day). Consider increasing transaction costs or hold incentives.{Color.END}")
        
    # Insight 3: Action Collapse
    collapsed_models = [m for m in models if m['max_action_pct'] > THRESHOLDS['action_collapse']]
    if collapsed_models:
        actions = [m['dominant_action'] for m in collapsed_models]
        most_common = Counter(actions).most_common(1)[0][0]
        insights.append(f"{Color.RED}CRITICAL: {len(collapsed_models)} models suffer from Action Collapse (>80% same action). Most common stuck action: {most_common}. Increase entropy coefficient or diversity bonuses.{Color.END}")
        
    # Insight 4: Algorithm Comparison
    algos = defaultdict(list)
    for m in models:
        algos[m['agent_type']].append(m)
        
    algo_stats = []
    for algo, m_list in algos.items():
        mean_ret = np.mean([m['return'] for m in m_list])
        mean_sharpe = np.mean([m['sharpe'] for m in m_list])
        algo_stats.append((algo, mean_ret, mean_sharpe))
        
    algo_stats.sort(key=lambda x: x[1], reverse=True)
    best_algo = algo_stats[0]
    insights.append(f"{Color.BLUE}OBSERVATION: {best_algo[0].upper()} is currently the best performing algorithm (Avg Return: {best_algo[1]:.2f}%, Sharpe: {best_algo[2]:.2f}).{Color.END}")
    
    # Insight 5: Winning vs Losing
    winning_models = [m for m in models if m['return'] > 0]
    win_rate = len(winning_models) / len(models)
    insights.append(f"Model Success Rate: {win_rate:.0%} of trained models are profitable.")

    return insights

def print_summary_of_findings(models: List[Dict[str, Any]]):

    """Print a structured summary of findings as requested."""

    if not models:

        return



    def format_algo(algo):

        if algo == "ppo": return "PPO"

        if algo == "recurrent_ppo": return "RecurrentPPO"

        if algo == "ensemble": return "Ensemble"

        return algo.upper()



    print(f"\n{Color.BOLD}Summary of Findings{Color.END}\n")

    print(f"It has scanned {len(models)} trained models and provided a detailed performance analysis.\n")

    

    # 1. Overall Health

    winning_models = [m for m in models if m['return'] > 0]

    win_rate = (len(winning_models) / len(models)) * 100 if models else 0

    avg_return = np.mean([m['return'] for m in models])

    

    health_status = "performing well overall" if avg_return > 10 else "showing mixed results"
    health_color = Color.GREEN if avg_return > 10 else Color.YELLOW
    if avg_return < 0: 
        health_status = "struggling"
        health_color = Color.RED

    

    print(f" * {Color.BOLD}Overall Health{Color.END}: The RL models are {health_color}{health_status}{Color.END}, with an {Color.CYAN}{win_rate:.0f}%{Color.END}")

    print(f"   profitability rate and an average return of {health_color}{avg_return:+.2f}%{Color.END}. ")

    

    # 2. Top Performers (Top 5 models)

    print(f" * {Color.BOLD}Top Performers{Color.END}:")

    best_performers = sorted(models, key=lambda x: x['return'], reverse=True)[:5]

            

    for m in best_performers:
        baseline_text = ""
        ret_color = Color.GREEN if m['return'] > 0 else Color.RED
        
        if m['buy_hold_return'] != 0.0:
            diff = m['return'] - m['buy_hold_return']
            sign = "+" if diff >= 0 else ""
            diff_color = Color.GREEN if diff >= 0 else Color.RED
            baseline_text = f" (vs B&H: {diff_color}{sign}{diff:.2f}%{Color.END})"

        print(f"     * {Color.CYAN}{m['symbol']}{Color.END} ({format_algo(m['agent_type'])}) [{m['name']}]: {ret_color}{m['return']:+.2f}% Return{Color.END}{baseline_text} (Sharpe {m['sharpe']:.2f})")

    # 3. Bottom Performers (show context, not just negative returns)
    worst_performers = sorted(models, key=lambda x: x['return'])[:5]

    if worst_performers:
        print(f" * {Color.BOLD}Bottom Performers{Color.END}:")

        for m in worst_performers:
            ret_color = Color.GREEN if m['return'] > 0 else Color.RED

            # Add market context if available
            context = ""
            if m['buy_hold_return'] != 0.0:
                outperformance = m['return'] - m['buy_hold_return']
                if outperformance > 0:
                    context = f" (beats market by {Color.GREEN}+{outperformance:.2f}%{Color.END})"
                else:
                    context = f" (underperforms market by {Color.RED}{outperformance:.2f}%{Color.END})"

            print(f"     * {Color.CYAN}{m['symbol']}{Color.END} ({format_algo(m['agent_type'])}) [{m['name']}]: {ret_color}{m['return']:+.2f}% Return{Color.END} (Sharpe {m['sharpe']:.2f}){context}")

def print_table(models: List[Dict[str, Any]]):
    """Print formatted table of model results."""
    # Columns: Symbol, Algo, Model Name, Return, Sharpe, MaxDD, WinRate, Actions(H, BS, BM, BL, SP, SA), Trades, Age
    print(f"\n{Color.BOLD}{'Symbol':<6} {'Algorithm':<14} {'Model Name':<45} {'Return':<10} {'Sharpe':<8} {'MaxDD':<9} {'WinRate':<9} {'HOLD':<6} {'BUY_S':<6} {'BUY_M':<6} {'BUY_L':<6} {'SELL_P':<7} {'SELL_A':<7} {'Trades':<8} {'Age':<8}{Color.END}")
    print("-" * 175)
    
    for m in models:
        # Color code return
        ret_str = f"{m['return']:+.2f}%"
        if m['return'] > 20: ret_color = Color.GREEN
        elif m['return'] > 0: ret_color = Color.CYAN
        elif m['return'] > -10: ret_color = Color.YELLOW
        else: ret_color = Color.RED
        
        # Color code Sharpe
        sharpe_val = m['sharpe']
        if sharpe_val > 1.5: sharpe_color = Color.GREEN
        elif sharpe_val > 0.5: sharpe_color = Color.CYAN
        elif sharpe_val > 0: sharpe_color = Color.YELLOW
        else: sharpe_color = Color.RED
        
        # Format Trades
        trades_str = str(m['trades'])
        if m['trades'] < 5: trades_str += " (L)" # Low
        elif m['trades'] > 100: trades_str += " (H)" # High
        
        # Calculate Action Percentages
        action_counts = m['actions']
        total = sum(action_counts.values())
        pcts = {k: (v/total*100) if total > 0 else 0 for k,v in action_counts.items()}
        
        # Format percentages as strings to avoid stray % signs in alignment
        ret_fmt = f"{ret_color}{ret_str:<10}{Color.END}"
        sharpe_fmt = f"{sharpe_color}{sharpe_val:<8.2f}{Color.END}"
        dd_fmt = f"{m['drawdown']*100:>6.1f}%"
        wr_fmt = f"{m['win_rate']*100:>6.0f}%"
        
        h_pct = f"{pcts.get('HOLD',0):>4.0f}%"
        bs_pct = f"{pcts.get('BUY_SMALL',0):>4.0f}%"
        bm_pct = f"{pcts.get('BUY_MEDIUM',0):>4.0f}%"
        bl_pct = f"{pcts.get('BUY_LARGE',0):>4.0f}%"
        sp_pct = f"{pcts.get('SELL_PARTIAL',0):>5.0f}%"
        sa_pct = f"{pcts.get('SELL_ALL',0):>5.0f}%"
        
        print(f"{m['symbol']:<6} {m['agent_type']:<14} {m['name']:<45} {ret_fmt} {sharpe_fmt} {dd_fmt:<9} {wr_fmt:<9} {h_pct:<6} {bs_pct:<6} {bm_pct:<6} {bl_pct:<6} {sp_pct:<7} {sa_pct:<7} {trades_str:<8} {m['age']:<8}")

def main():
    parser = argparse.ArgumentParser(description="Analyze RL training results and provide insights.")
    parser.add_argument("--symbol", type=str, help="Filter by stock symbol")
    parser.add_argument("--min-trades", type=int, default=0, help="Filter models with fewer than N trades")
    parser.add_argument("--sort", type=str, choices=['return', 'sharpe', 'date'], default='date', help="Sort order")
    parser.add_argument("--prune", action="store_true", help="Move models matching criteria to archive")
    parser.add_argument("--min-return", type=float, default=0.0, help="Prune models with return less than this value")
    parser.add_argument("--age", type=str, default="24h", help="Prune models older than this (e.g. '24h', '7d')")
    args = parser.parse_args()
    
    print(f"\n{Color.HEADER}{'='*80}")
    print(f"🧠 RL TRAINING EVALUATION & INSIGHTS")
    print(f"{Color.HEADER}{'='*80}{Color.END}")
    
    # 1. Scan for models
    if not MODELS_DIR.exists():
        print(f"{Color.RED}Error: Models directory {MODELS_DIR} not found.{Color.END}")
        return

    model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name != 'archive' and (d / "backtest_results.json").exists()]
    
    if not model_dirs:
        print(f"{Color.YELLOW}No trained models found in {MODELS_DIR}.{Color.END}")
        return
        
    print(f"Scanning {len(model_dirs)} models...")
    
    # 2. Analyze models
    analyzed_models = []
    for d in model_dirs:
        data = analyze_model(d)
        if data:
            # Apply filters
            if args.symbol and data['symbol'].upper() != args.symbol.upper():
                continue
            if data['trades'] < args.min_trades:
                continue
            analyzed_models.append(data)
            
    if not analyzed_models:
        print(f"No models matched your criteria.")
        return

    # Sort models
    if args.sort == 'return':
        analyzed_models.sort(key=lambda x: x['return'], reverse=True)
    elif args.sort == 'sharpe':
        analyzed_models.sort(key=lambda x: x['sharpe'], reverse=True)
    else: # date
        # Sort by mtime (newest first)
        analyzed_models.sort(key=lambda x: x['path'].stat().st_mtime, reverse=True)

    # 3. Print Detailed Table
    print_table(analyzed_models)
    
    # 4. Handle Pruning
    if args.prune:
        age_seconds = parse_age_to_seconds(args.age)
        prune_models(analyzed_models, args.min_return, age_seconds)
        # Exit after pruning - user can re-run to see updated state
        return
    
    # 5. Print Summary of Findings
    print_summary_of_findings(analyzed_models)
    
    # 6. Generate Insights
    print(f"\n{Color.HEADER}{'='*80}")
    print(f"💡 TRAINING INSIGHTS")
    print(f"{Color.HEADER}{'='*80}{Color.END}")
    
    insights = generate_insights(analyzed_models)
    for insight in insights:
        print(f"• {insight}")
        
    print("\n")

if __name__ == "__main__":
    main()
