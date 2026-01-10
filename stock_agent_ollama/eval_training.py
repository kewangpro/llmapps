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
LSTM_MODELS_DIR = Path("data/models/lstm")
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
    """Get readable age based on when backtest_results.json was last modified."""
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

def prune_models(models: List[Dict[str, Any]], min_return: float, age_seconds: int, keep_best: bool = False):
    """Archive models matching criteria."""
    mode_desc = "Keep Best per Symbol/Type" if keep_best else f"Return < {min_return}%"
    print(f"\n{Color.HEADER}{'='*80}")
    print(f"🧹 PRUNING MODELS ({mode_desc}, Age > {age_seconds/3600:.1f}h)")
    print(f"{Color.HEADER}{'='*80}{Color.END}")
    
    threshold_time = datetime.now().timestamp() - age_seconds
    to_prune = []
    
    if keep_best:
        # Group models by symbol and agent type
        grouped_models = defaultdict(lambda: defaultdict(list))
        for m in models:
            grouped_models[m['symbol']][m['agent_type']].append(m)

        for symbol, agent_types in grouped_models.items():
            for agent_type, model_list in agent_types.items():
                # Sort by return (descending)
                model_list.sort(key=lambda x: x['return'], reverse=True)
                
                # Best model is the first one
                best_model = model_list[0]
                print(f"  Keeping best {symbol} {agent_type}: {best_model['name']} (Ret: {best_model['return']:.2f}%)")
                
                # The rest are candidates for pruning
                candidates = model_list[1:]
                for m in candidates:
                    if m['mtime'] < threshold_time:
                        to_prune.append(m)
                    else:
                        print(f"    Skipping {m['name']} (not best, but too new: {m['age']})")
    else:
        for m in models:
            # We prune models that are losing money AND older than the threshold
            if m['return'] < min_return and m['mtime'] < threshold_time:
                to_prune.append(m)
            
    if not to_prune:
        print("\nNo models matched the pruning criteria.")
        return
        
    if not ARCHIVE_DIR.exists():
        ARCHIVE_DIR.mkdir(parents=True)
        print(f"Created archive directory: {ARCHIVE_DIR}")
        
    print(f"\nFound {len(to_prune)} models to archive...\n")
    
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

def analyze_lstm_model(metadata_path: Path) -> Dict[str, Any]:
    """Analyze a single LSTM model from its metadata."""
    if not metadata_path.exists():
        return None

    metadata = load_json(metadata_path)
    if not metadata:
        return None

    symbol = metadata.get('symbol', 'Unknown')
    training_date = metadata.get('training_date', '')

    # Parse training date
    try:
        train_dt = datetime.fromisoformat(training_date)
        delta = datetime.now() - train_dt
        if delta.days > 0:
            age = f"{delta.days}d ago"
        elif delta.seconds > 3600:
            age = f"{delta.seconds // 3600}h ago"
        else:
            age = f"{delta.seconds // 60}m ago"
        mtime = train_dt.timestamp()
    except:
        age = "?"
        mtime = metadata_path.stat().st_mtime

    # Extract training histories
    histories = metadata.get('training_histories', [])

    # Calculate average final loss and val_loss across ensemble
    final_losses = []
    final_val_losses = []
    final_maes = []
    final_val_maes = []

    for hist in histories:
        if 'loss' in hist and len(hist['loss']) > 0:
            final_losses.append(hist['loss'][-1])
        if 'val_loss' in hist and len(hist['val_loss']) > 0:
            final_val_losses.append(hist['val_loss'][-1])
        if 'mean_absolute_error' in hist and len(hist['mean_absolute_error']) > 0:
            final_maes.append(hist['mean_absolute_error'][-1])
        if 'val_mean_absolute_error' in hist and len(hist['val_mean_absolute_error']) > 0:
            final_val_maes.append(hist['val_mean_absolute_error'][-1])

    avg_final_loss = np.mean(final_losses) if final_losses else 0.0
    avg_final_val_loss = np.mean(final_val_losses) if final_val_losses else 0.0
    avg_final_mae = np.mean(final_maes) if final_maes else 0.0
    avg_final_val_mae = np.mean(final_val_maes) if final_val_maes else 0.0

    # Model architecture info
    arch = metadata.get('model_architecture', {})
    feature_count = arch.get('feature_count', 5)
    uses_enhanced = arch.get('uses_enhanced_features', False)

    # Calculate overfitting metric (val_loss / train_loss ratio)
    overfit_ratio = (avg_final_val_loss / avg_final_loss) if avg_final_loss > 0 else 1.0

    return {
        'symbol': symbol,
        'age': age,
        'mtime': mtime,
        'ensemble_size': metadata.get('ensemble_size', 3),
        'final_loss': avg_final_loss,
        'final_val_loss': avg_final_val_loss,
        'final_mae': avg_final_mae,
        'final_val_mae': avg_final_val_mae,
        'overfit_ratio': overfit_ratio,
        'feature_count': feature_count,
        'uses_enhanced_features': uses_enhanced,
        'sequence_length': metadata.get('sequence_length', 90),
        'path': metadata_path,
        'model_type': 'LSTM'
    }

def scan_lstm_models() -> List[Dict[str, Any]]:
    """Scan and analyze all LSTM models."""
    if not LSTM_MODELS_DIR.exists():
        return []

    lstm_models = []
    metadata_files = list(LSTM_MODELS_DIR.glob("*_metadata.json"))

    # Filter out horizon-specific models (e.g., *_h30_metadata.json)
    metadata_files = [f for f in metadata_files if '_h' not in f.stem or f.stem.endswith('_metadata')]

    for metadata_path in metadata_files:
        # Skip horizon-specific models (those with _hN_ pattern)
        if '_h' in metadata_path.stem and not metadata_path.stem.endswith('_metadata'):
            continue

        data = analyze_lstm_model(metadata_path)
        if data:
            lstm_models.append(data)

    return lstm_models

def generate_lstm_insights(models: List[Dict[str, Any]]) -> List[str]:
    """Generate insights specific to LSTM models."""
    insights = []

    if not models:
        return ["No LSTM models found."]

    avg_val_loss = np.mean([m['final_val_loss'] for m in models])
    avg_mae = np.mean([m['final_val_mae'] for m in models])

    # Insight 1: Overall LSTM Performance
    if avg_val_loss < 0.05:
        insights.append(f"{Color.GREEN}EXCELLENT: LSTM models show very low validation loss (Avg: {avg_val_loss:.4f}). Predictions are highly accurate.{Color.END}")
    elif avg_val_loss < 0.15:
        insights.append(f"{Color.CYAN}GOOD: LSTM models show acceptable validation loss (Avg: {avg_val_loss:.4f}). MAE: {avg_mae:.4f}.{Color.END}")
    else:
        insights.append(f"{Color.YELLOW}WARNING: LSTM models have high validation loss (Avg: {avg_val_loss:.4f}). Consider retraining with more data or different features.{Color.END}")

    # Insight 2: Overfitting Detection
    overfit_models = [m for m in models if m['overfit_ratio'] > 1.5]
    if len(overfit_models) > len(models) * 0.3:
        insights.append(f"{Color.RED}CRITICAL: {len(overfit_models)} LSTM models ({len(overfit_models)/len(models):.0%}) show signs of overfitting (val_loss/train_loss > 1.5). Increase regularization or reduce model complexity.{Color.END}")

    # Insight 3: Feature Usage
    enhanced_models = [m for m in models if m['uses_enhanced_features']]
    if enhanced_models:
        avg_val_loss_enhanced = np.mean([m['final_val_loss'] for m in enhanced_models])
        basic_models = [m for m in models if not m['uses_enhanced_features']]
        if basic_models:
            avg_val_loss_basic = np.mean([m['final_val_loss'] for m in basic_models])
            if avg_val_loss_enhanced < avg_val_loss_basic:
                improvement = ((avg_val_loss_basic - avg_val_loss_enhanced) / avg_val_loss_basic) * 100
                insights.append(f"{Color.GREEN}POSITIVE: Enhanced features improve LSTM performance by {improvement:.1f}% (Val Loss: {avg_val_loss_enhanced:.4f} vs {avg_val_loss_basic:.4f}).{Color.END}")
        insights.append(f"Feature Usage: {len(enhanced_models)}/{len(models)} models use enhanced features.")

    return insights

def print_lstm_table(models: List[Dict[str, Any]]):
    """Print formatted table of LSTM model results."""
    print(f"\n{Color.BOLD}{'Symbol':<8} {'Ensemble':<9} {'Features':<10} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<12} {'Val MAE':<12} {'Overfit':<9} {'Age':<10}{Color.END}")
    print("-" * 120)

    for m in models:
        # Color code validation loss
        val_loss = m['final_val_loss']
        if val_loss < 0.05:
            val_color = Color.GREEN
        elif val_loss < 0.15:
            val_color = Color.CYAN
        elif val_loss < 0.30:
            val_color = Color.YELLOW
        else:
            val_color = Color.RED

        # Color code overfitting
        overfit = m['overfit_ratio']
        if overfit < 1.2:
            overfit_color = Color.GREEN
        elif overfit < 1.5:
            overfit_color = Color.YELLOW
        else:
            overfit_color = Color.RED

        feature_type = f"Enh({m['feature_count']})" if m['uses_enhanced_features'] else "Basic(5)"

        print(f"{m['symbol']:<8} {m['ensemble_size']:<9} {feature_type:<10} "
              f"{m['final_loss']:<12.4f} {val_color}{m['final_val_loss']:<12.4f}{Color.END} "
              f"{m['final_mae']:<12.4f} {m['final_val_mae']:<12.4f} "
              f"{overfit_color}{overfit:<9.2f}{Color.END} {m['age']:<10}")

def print_combined_rankings(rl_models: List[Dict[str, Any]], lstm_models: List[Dict[str, Any]]):
    """Print combined rankings showing both RL and LSTM performance for each symbol."""
    print(f"\n{Color.HEADER}{'='*120}")
    print(f"📊 COMBINED MODEL RANKINGS (RL + LSTM)")
    print(f"{Color.HEADER}{'='*120}{Color.END}")

    # Group by symbol
    symbols = set([m['symbol'] for m in rl_models] + [m['symbol'] for m in lstm_models])

    combined = []
    for symbol in symbols:
        # Get best RL model for this symbol
        symbol_rl_models = [m for m in rl_models if m['symbol'] == symbol]
        best_rl = max(symbol_rl_models, key=lambda x: x['return']) if symbol_rl_models else None

        # Get LSTM model for this symbol
        symbol_lstm = next((m for m in lstm_models if m['symbol'] == symbol), None)

        combined.append({
            'symbol': symbol,
            'rl_model': best_rl,
            'lstm_model': symbol_lstm,
            'has_rl': best_rl is not None,
            'has_lstm': symbol_lstm is not None,
            'rl_return': best_rl['return'] if best_rl else None,
            'rl_sharpe': best_rl['sharpe'] if best_rl else None,
            'lstm_val_loss': symbol_lstm['final_val_loss'] if symbol_lstm else None,
            'lstm_val_mae': symbol_lstm['final_val_mae'] if symbol_lstm else None
        })

    # Sort by RL performance (symbols with both models first)
    combined.sort(key=lambda x: (
        not (x['has_rl'] and x['has_lstm']),
        -(x['rl_return'] or -1000),
        x['lstm_val_loss'] or 999
    ))

    # Print header
    print(f"\n{Color.BOLD}{'Symbol':<8} {'RL Models':<12} {'Best Return':<13} {'Sharpe':<8} {'LSTM Status':<13} {'Val Loss':<11} {'Val MAE':<11} {'Recommendation':<30}{Color.END}")
    print("-" * 120)

    for item in combined:
        symbol = item['symbol']

        # RL info
        if item['has_rl']:
            rl_count = len([m for m in rl_models if m['symbol'] == symbol])
            rl_models_str = f"{rl_count} models"

            ret = item['rl_return']
            ret_color = Color.GREEN if ret > 20 else (Color.CYAN if ret > 0 else Color.RED)
            ret_str = f"{ret_color}{ret:+.2f}%{Color.END}"

            sharpe = item['rl_sharpe']
            sharpe_color = Color.GREEN if sharpe > 1.5 else (Color.CYAN if sharpe > 0.5 else Color.YELLOW)
            sharpe_str = f"{sharpe_color}{sharpe:.2f}{Color.END}"
        else:
            rl_models_str = "None"
            ret_str = "-"
            sharpe_str = "-"

        # LSTM info
        if item['has_lstm']:
            lstm_status = "Trained"
            val_loss = item['lstm_val_loss']
            val_loss_color = Color.GREEN if val_loss < 0.05 else (Color.CYAN if val_loss < 0.15 else Color.YELLOW)
            val_loss_str = f"{val_loss_color}{val_loss:.4f}{Color.END}"
            val_mae_str = f"{item['lstm_val_mae']:.4f}"
        else:
            lstm_status = "Not trained"
            val_loss_str = "-"
            val_mae_str = "-"

        # Generate recommendation
        recommendation = ""
        if item['has_rl'] and item['has_lstm']:
            if item['rl_return'] > 10 and item['lstm_val_loss'] < 0.10:
                recommendation = f"{Color.GREEN}✓ Ready for live trading{Color.END}"
            elif item['rl_return'] > 0 and item['lstm_val_loss'] < 0.20:
                recommendation = f"{Color.CYAN}Good for testing{Color.END}"
            elif item['rl_return'] < 0:
                recommendation = f"{Color.YELLOW}⚠ RL needs retraining{Color.END}"
            elif item['lstm_val_loss'] > 0.30:
                recommendation = f"{Color.YELLOW}⚠ LSTM needs retraining{Color.END}"
            else:
                recommendation = "Monitor performance"
        elif item['has_rl'] and not item['has_lstm']:
            recommendation = f"{Color.YELLOW}Train LSTM model{Color.END}"
        elif not item['has_rl'] and item['has_lstm']:
            recommendation = f"{Color.YELLOW}Train RL agent{Color.END}"
        else:
            recommendation = f"{Color.RED}Train both models{Color.END}"

        print(f"{symbol:<8} {rl_models_str:<12} {ret_str:<22} {sharpe_str:<17} {lstm_status:<13} {val_loss_str:<20} {val_mae_str:<11} {recommendation:<45}")

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
    parser = argparse.ArgumentParser(description="Analyze RL and LSTM training results and provide insights.")
    parser.add_argument("--symbol", type=str, help="Filter by stock symbol")
    parser.add_argument("--min-trades", type=int, default=0, help="Filter RL models with fewer than N trades")
    parser.add_argument("--sort", type=str, choices=['return', 'sharpe', 'winrate', 'maxdd', 'age'], default='age', help="Sort order for RL models")
    parser.add_argument("--prune", action="store_true", help="Move RL models matching criteria to archive")
    parser.add_argument("--keep-best", action="store_true", help="When pruning, keep the best model for each symbol/type pair")
    parser.add_argument("--min-return", type=float, default=0.0, help="Prune models with return less than this value (ignored if --keep-best is used)")
    parser.add_argument("--age", type=str, default="24h", help="Prune models older than this (e.g. '24h', '7d')")
    parser.add_argument("--rl-only", action="store_true", help="Show only RL models (skip LSTM)")
    parser.add_argument("--lstm-only", action="store_true", help="Show only LSTM models (skip RL)")
    parser.add_argument("--combined", action="store_true", help="Show combined rankings (default if both model types exist)")
    args = parser.parse_args()

    print(f"\n{Color.HEADER}{'='*120}")
    print(f"🧠 MODEL TRAINING EVALUATION & INSIGHTS (RL + LSTM)")
    print(f"{Color.HEADER}{'='*120}{Color.END}")

    # 1. Scan for RL models
    analyzed_rl_models = []
    if not args.lstm_only:
        if not MODELS_DIR.exists():
            print(f"{Color.YELLOW}Warning: RL models directory {MODELS_DIR} not found.{Color.END}")
        else:
            model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name != 'archive' and (d / "backtest_results.json").exists()]

            if model_dirs:
                print(f"Scanning {len(model_dirs)} RL models...")

                for d in model_dirs:
                    data = analyze_model(d)
                    if data:
                        # Apply filters
                        if args.symbol and data['symbol'].upper() != args.symbol.upper():
                            continue
                        if data['trades'] < args.min_trades:
                            continue
                        analyzed_rl_models.append(data)

                # Sort RL models
                if args.sort == 'return':
                    analyzed_rl_models.sort(key=lambda x: x['return'], reverse=True)
                elif args.sort == 'sharpe':
                    analyzed_rl_models.sort(key=lambda x: x['sharpe'], reverse=True)
                elif args.sort == 'winrate':
                    analyzed_rl_models.sort(key=lambda x: x['win_rate'], reverse=True)
                elif args.sort == 'maxdd':
                    analyzed_rl_models.sort(key=lambda x: abs(x['drawdown']))
                else: # age
                    analyzed_rl_models.sort(key=lambda x: x['mtime'], reverse=True)
            else:
                print(f"{Color.YELLOW}No RL models found in {MODELS_DIR}.{Color.END}")

    # 2. Scan for LSTM models
    analyzed_lstm_models = []
    if not args.rl_only:
        print(f"Scanning LSTM models...")
        analyzed_lstm_models = scan_lstm_models()

        # Apply symbol filter if specified
        if args.symbol:
            analyzed_lstm_models = [m for m in analyzed_lstm_models if m['symbol'].upper() == args.symbol.upper()]

        # Sort LSTM models by age (newest first)
        analyzed_lstm_models.sort(key=lambda x: x['mtime'], reverse=True)

        if analyzed_lstm_models:
            print(f"Found {len(analyzed_lstm_models)} LSTM models")
        else:
            print(f"{Color.YELLOW}No LSTM models found in {LSTM_MODELS_DIR}.{Color.END}")

    # 3. Handle Pruning (RL only)
    if args.prune:
        if not analyzed_rl_models:
            print(f"{Color.YELLOW}No RL models to prune.{Color.END}")
            return
        age_seconds = parse_age_to_seconds(args.age)
        prune_models(analyzed_rl_models, args.min_return, age_seconds, keep_best=args.keep_best)
        return

    # 4. Display Results
    if not analyzed_rl_models and not analyzed_lstm_models:
        print(f"{Color.YELLOW}No models found matching your criteria.{Color.END}")
        return

    # Display RL models
    if analyzed_rl_models:
        print(f"\n{Color.HEADER}{'='*120}")
        print(f"📈 REINFORCEMENT LEARNING MODELS ({len(analyzed_rl_models)} models)")
        print(f"{Color.HEADER}{'='*120}{Color.END}")
        print_table(analyzed_rl_models)
        print_summary_of_findings(analyzed_rl_models)

        print(f"\n{Color.HEADER}{'='*120}")
        print(f"💡 RL TRAINING INSIGHTS")
        print(f"{Color.HEADER}{'='*120}{Color.END}")
        rl_insights = generate_insights(analyzed_rl_models)
        for insight in rl_insights:
            print(f"• {insight}")

    # Display LSTM models
    if analyzed_lstm_models:
        print(f"\n{Color.HEADER}{'='*120}")
        print(f"🧬 LSTM PREDICTION MODELS ({len(analyzed_lstm_models)} models)")
        print(f"{Color.HEADER}{'='*120}{Color.END}")
        print_lstm_table(analyzed_lstm_models)

        print(f"\n{Color.HEADER}{'='*120}")
        print(f"💡 LSTM TRAINING INSIGHTS")
        print(f"{Color.HEADER}{'='*120}{Color.END}")
        lstm_insights = generate_lstm_insights(analyzed_lstm_models)
        for insight in lstm_insights:
            print(f"• {insight}")

    # Display Combined Rankings
    if analyzed_rl_models and analyzed_lstm_models and not args.rl_only and not args.lstm_only:
        print_combined_rankings(analyzed_rl_models, analyzed_lstm_models)

        # Combined insights
        print(f"\n{Color.HEADER}{'='*120}")
        print(f"🎯 COMBINED INSIGHTS & RECOMMENDATIONS")
        print(f"{Color.HEADER}{'='*120}{Color.END}")

        # Count symbols with both models
        rl_symbols = set([m['symbol'] for m in analyzed_rl_models])
        lstm_symbols = set([m['symbol'] for m in analyzed_lstm_models])
        both_symbols = rl_symbols & lstm_symbols

        print(f"• Total Symbols: {len(rl_symbols | lstm_symbols)}")
        print(f"  - With both RL and LSTM: {len(both_symbols)}")
        print(f"  - RL only: {len(rl_symbols - lstm_symbols)}")
        print(f"  - LSTM only: {len(lstm_symbols - rl_symbols)}")

        # Identify top candidates for live trading
        ready_for_live = []
        for symbol in both_symbols:
            best_rl = max([m for m in analyzed_rl_models if m['symbol'] == symbol], key=lambda x: x['return'])
            lstm = next(m for m in analyzed_lstm_models if m['symbol'] == symbol)

            if best_rl['return'] > 10 and best_rl['sharpe'] > 1.0 and lstm['final_val_loss'] < 0.10:
                ready_for_live.append({
                    'symbol': symbol,
                    'rl_return': best_rl['return'],
                    'rl_sharpe': best_rl['sharpe'],
                    'lstm_val_loss': lstm['final_val_loss']
                })

        if ready_for_live:
            ready_for_live.sort(key=lambda x: x['rl_return'], reverse=True)
            print(f"\n{Color.GREEN}✓ Top Candidates for Live Trading:{Color.END}")
            for item in ready_for_live[:5]:
                print(f"  {Color.BOLD}{item['symbol']}{Color.END}: RL Return {item['rl_return']:+.2f}%, Sharpe {item['rl_sharpe']:.2f}, LSTM Val Loss {item['lstm_val_loss']:.4f}")

        # Identify symbols needing attention
        needs_work = []
        for symbol in both_symbols:
            best_rl = max([m for m in analyzed_rl_models if m['symbol'] == symbol], key=lambda x: x['return'])
            lstm = next(m for m in analyzed_lstm_models if m['symbol'] == symbol)

            if best_rl['return'] < 0 or lstm['final_val_loss'] > 0.30:
                needs_work.append({
                    'symbol': symbol,
                    'issue': 'Poor RL performance' if best_rl['return'] < 0 else 'High LSTM loss',
                    'rl_return': best_rl['return'],
                    'lstm_val_loss': lstm['final_val_loss']
                })

        if needs_work:
            print(f"\n{Color.YELLOW}⚠ Symbols Needing Retraining:{Color.END}")
            for item in needs_work[:5]:
                print(f"  {Color.BOLD}{item['symbol']}{Color.END}: {item['issue']} (RL: {item['rl_return']:+.2f}%, LSTM: {item['lstm_val_loss']:.4f})")

    print("\n")

if __name__ == "__main__":
    main()
