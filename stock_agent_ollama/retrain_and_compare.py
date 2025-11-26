#!/usr/bin/env python3
"""
RL Model Retraining and Comparison Script

Usage:
    python retrain_and_compare.py --symbol SYMBOL [options]

Options:
    --symbol SYMBOL         Stock symbol to train on (required)
    --algorithms ALG1,ALG2  Comma-separated list of algorithms to train
                           (options: ppo, recurrent_ppo, a2c, qrdqn)
                           (default: all)
    --timesteps N          Training timesteps (default: 300000)
    --skip-training        Skip training, only run backtest
    --no-baselines         Skip baseline strategies in backtest

Example:
    # Retrain all algorithms on TSLA
    python retrain_and_compare.py --symbol TSLA

    # Retrain only A2C and QRDQN on AAPL
    python retrain_and_compare.py --symbol AAPL --algorithms a2c,qrdqn

    # Only backtest existing models for NVDA
    python retrain_and_compare.py --symbol NVDA --skip-training
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl import EnhancedRLTrainer, EnhancedTrainingConfig
from src.rl import BacktestEngine, BacktestConfig
from src.rl.baselines import BuyHoldStrategy, MomentumStrategy
from src.rl.model_utils import load_rl_agent, load_env_config_from_model
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(symbol: str, agent_type: str, total_timesteps: int = 300000) -> Path:
    """
    Train a single RL model.

    Args:
        symbol: Stock symbol
        agent_type: Algorithm type (ppo, recurrent_ppo, a2c, sac, qrdqn)
        total_timesteps: Total training timesteps

    Returns:
        Path to saved model directory
    """
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {agent_type.upper()} on {symbol}")
    print(f"{'='*80}\n")

    # Setup dates (3 years of training data)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")

    # Create training config
    config = EnhancedTrainingConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        agent_type=agent_type,
        total_timesteps=total_timesteps,
        learning_rate=0.0003,
        enable_diagnostics=True
    )

    # Train
    try:
        trainer = EnhancedRLTrainer(config)
        results = trainer.train()

        print(f"\n✅ {agent_type.upper()} Training Complete!")
        print(f"   Model saved to: {trainer.save_dir}")
        return trainer.save_dir

    except Exception as e:
        logger.error(f"Training failed for {agent_type}: {e}")
        raise


def find_latest_model(symbol: str, agent_type: str) -> Path:
    """
    Find the most recently trained model.

    Args:
        symbol: Stock symbol
        agent_type: Algorithm type

    Returns:
        Path to model file, or None if not found
    """
    models_dir = Path("data/models/rl")
    if not models_dir.exists():
        return None

    # Find all model directories for this symbol and agent type
    pattern = f"{agent_type.lower()}_{symbol}_*"
    matching_dirs = sorted(
        models_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not matching_dirs:
        return None

    latest_dir = matching_dirs[0]
    model_path = latest_dir / "best_model.zip"
    if not model_path.exists():
        model_path = latest_dir / "final_model.zip"

    return model_path if model_path.exists() else None


def run_comprehensive_backtest(symbol: str, include_baselines: bool = True) -> dict:
    """
    Run backtest on all available models.

    Args:
        symbol: Stock symbol
        include_baselines: Whether to include baseline strategies

    Returns:
        Dictionary of strategy name -> backtest result
    """
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE BACKTEST: {symbol}")
    print(f"{'='*80}\n")

    # Setup backtest period (last 6 months + buffer)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=280)).strftime("%Y-%m-%d")

    results = {}

    # Test all algorithms
    algorithms = ['ppo', 'recurrent_ppo', 'a2c', 'sac', 'qrdqn']

    for agent_type in algorithms:
        logger.info(f"Backtesting {agent_type.upper()}...")
        model_path = find_latest_model(symbol, agent_type)

        if not model_path:
            logger.warning(f"No model found for {agent_type}")
            continue

        try:
            # Load training config
            training_config = load_env_config_from_model(model_path)

            # Determine include_trend_indicators
            include_trend = training_config.get('include_trend_indicators', False)
            if agent_type == 'recurrent_ppo':
                include_trend = True

            # Create backtest config
            backtest_config = BacktestConfig(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                use_improved_actions=training_config.get('use_improved_actions', True),
                include_trend_indicators=include_trend
            )

            # Create engine and run backtest
            engine = BacktestEngine(backtest_config)
            agent = load_rl_agent(model_path, env=None)

            result = engine.run_agent_backtest(agent, deterministic=True)

            # Store result
            display_name = agent_type.upper().replace('_', ' ')
            results[f"{display_name} Agent"] = result

            print(f"   ✅ {display_name}: {result.metrics.total_return_pct:+.2f}%")

        except Exception as e:
            logger.error(f"Failed to backtest {agent_type}: {e}", exc_info=True)

    # Run baseline strategies
    if include_baselines:
        try:
            default_config = BacktestConfig(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            baseline_engine = BacktestEngine(default_config)

            # Buy & Hold
            buy_hold = BuyHoldStrategy()
            buy_hold_result = baseline_engine.run_strategy_backtest(buy_hold.get_action)
            results["Buy & Hold"] = buy_hold_result
            print(f"   ✅ Buy & Hold: {buy_hold_result.metrics.total_return_pct:+.2f}%")

            # Momentum
            momentum = MomentumStrategy()
            momentum_result = baseline_engine.run_strategy_backtest(momentum.get_action)
            results["Momentum"] = momentum_result
            print(f"   ✅ Momentum: {momentum_result.metrics.total_return_pct:+.2f}%")

        except Exception as e:
            logger.error(f"Failed to run baseline strategies: {e}", exc_info=True)

    return results


def display_results(results: dict):
    """
    Display backtest results in a formatted table.

    Args:
        results: Dictionary of strategy name -> backtest result
    """
    print(f"\n{'='*80}")
    print("📊 BACKTEST COMPARISON")
    print(f"{'='*80}\n")

    if not results:
        print("No results to display")
        return

    # Create results DataFrame
    data = []
    for strategy, result in results.items():
        metrics = result.metrics
        actions = result.action_counts

        # Calculate action percentages
        total_actions = sum(actions.values())
        action_pcts = {
            k: (v / total_actions * 100) if total_actions > 0 else 0
            for k, v in actions.items()
        }

        data.append({
            'Strategy': strategy,
            'Return': f"{metrics.total_return_pct:+.2f}%",
            'Sharpe': f"{metrics.sharpe_ratio:.2f}",
            'Max DD': f"{metrics.max_drawdown_pct:.2f}%",
            'Win Rate': f"{metrics.win_rate:.0f}%",
            'HOLD': f"{action_pcts.get('HOLD', 0):.0f}%",
            'BUY_S': f"{action_pcts.get('BUY_SMALL', 0):.0f}%",
            'BUY_M': f"{action_pcts.get('BUY_MEDIUM', 0):.0f}%",
            'BUY_L': f"{action_pcts.get('BUY_LARGE', 0):.0f}%",
            'SELL_P': f"{action_pcts.get('SELL_PARTIAL', 0):.0f}%",
            'SELL_A': f"{action_pcts.get('SELL_ALL', 0):.0f}%",
            'Trades': metrics.total_trades,
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()


def analyze_results(results: dict):
    """
    Provide analysis and recommendations based on backtest results.

    Args:
        results: Dictionary of strategy name -> backtest result
    """
    print("=" * 80)
    print("🎯 ANALYSIS")
    print("=" * 80)
    print()

    if not results:
        print("No results to analyze")
        return

    # Find best performer by return
    best_strategy = max(
        results.items(),
        key=lambda x: x[1].metrics.total_return_pct
    )
    best_name, best_result = best_strategy

    print(f"🥇 Best Performer: {best_name}")
    print(f"   Return: {best_result.metrics.total_return_pct:+.2f}%")
    print(f"   Sharpe: {best_result.metrics.sharpe_ratio:.2f}")

    # Check for action collapse in RL agents
    print(f"\n📊 Action Distribution Analysis:")
    for strategy, result in results.items():
        if "Agent" in strategy:
            actions = result.action_counts
            total = sum(actions.values())

            if total > 0:
                max_action = max(actions.values())
                max_pct = (max_action / total) * 100

                if max_pct > 80:
                    print(f"   ⚠️  {strategy}: ACTION COLLAPSE ({max_pct:.0f}% single action)")
                else:
                    print(f"   ✅ {strategy}: Balanced actions (max {max_pct:.0f}%)")

    print()
    print("=" * 80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain RL models and compare performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Stock symbol to train on (required)'
    )

    parser.add_argument(
        '--algorithms',
        type=str,
        default='all',
        help='Comma-separated algorithms to train (default: all)'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=300000,
        help='Training timesteps (default: 300000)'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training, only run backtest'
    )

    parser.add_argument(
        '--no-baselines',
        action='store_true',
        help='Skip baseline strategies in backtest'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + " "*25 + "RL MODEL RETRAINING" + " "*34 + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}\n")

    print(f"Symbol: {args.symbol}")
    print(f"Timesteps: {args.timesteps:,}")

    # Determine which algorithms to train
    if args.algorithms == 'all':
        algorithms = ['ppo', 'recurrent_ppo', 'a2c', 'sac', 'qrdqn']
    else:
        algorithms = [a.strip().lower() for a in args.algorithms.split(',')]

    print(f"Algorithms: {', '.join([a.upper() for a in algorithms])}")
    print()

    # Training phase
    if not args.skip_training:
        for i, agent_type in enumerate(algorithms, 1):
            print(f"\n📌 Step {i}/{len(algorithms)}: Training {agent_type.upper()}")
            try:
                train_model(args.symbol, agent_type, args.timesteps)
            except Exception as e:
                logger.error(f"Skipping {agent_type} due to error: {e}")
                continue
    else:
        print("⏭️  Skipping training phase")

    # Backtesting phase
    print(f"\n📌 Running Comprehensive Backtest")
    results = run_comprehensive_backtest(
        args.symbol,
        include_baselines=not args.no_baselines
    )

    # Display and analyze results
    display_results(results)
    analyze_results(results)

    print("✨ Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
