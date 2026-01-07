#!/usr/bin/env python3
"""
RL Model Training Script

This script focuses solely on training RL agents. 
It automatically triggers a validation backtest upon completion to generate results.

Usage:
    python retrain_rl.py (--symbol SYMBOL | --watchlist) [options]

Options:
    --symbol SYMBOL         Stock symbol(s) to train on - single or comma-separated list
    --watchlist             Train on all symbols from default watchlist
    --algorithms ALG1,ALG2  Comma-separated list of algorithms to train
                           (options: ppo, recurrent_ppo, ensemble)
                           (default: all)
    --timesteps N          Training timesteps (default: 300000)

Note: Either --symbol or --watchlist is required

Examples:
    # Train all algorithms on TSLA
    python retrain_rl.py --symbol TSLA

    # Train PPO and Ensemble on AAPL
    python retrain_rl.py --symbol AAPL --algorithms ppo,ensemble

    # Train all watchlist symbols with all algorithms
    python retrain_rl.py --watchlist
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl import EnhancedRLTrainer, EnhancedTrainingConfig
from src.tools.portfolio_manager import portfolio_manager
import validate_backtest  # Import validation script as module

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
        agent_type: Algorithm type (ppo, recurrent_ppo, ensemble)
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Create mutually exclusive group for symbol/watchlist
    symbol_group = parser.add_mutually_exclusive_group(required=True)

    symbol_group.add_argument(
        '--symbol',
        type=str,
        help='Stock symbol(s) to train on - single symbol or comma-separated list'
    )

    symbol_group.add_argument(
        '--watchlist',
        action='store_true',
        help='Train on all symbols from default watchlist'
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

    # Deprecated args handling (for backward compat)
    parser.add_argument('--skip-training', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-baselines', action='store_true', help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.skip_training:
        print("⚠️  The --skip-training flag is deprecated in this script.")
        print("   Please use 'python eval_training.py' or 'python validate_backtest.py' for analysis.")
        sys.exit(1)

    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + " "*25 + "RL MODEL TRAINING" + " "*36 + "#")
    print("#" + " "*78 + "#")
    print(f"{ '#'*80}\n")

    # Determine symbols to process
    if args.watchlist:
        symbols = portfolio_manager.load_portfolio("default")
        if not symbols:
            print("❌ Error: Watchlist is empty or not found")
            sys.exit(1)
        print(f"Using watchlist symbols: {', '.join(symbols)}")
    else:
        symbols = [s.strip().upper() for s in args.symbol.split(',')]
        print(f"Symbol(s): {', '.join(symbols)}")

    print(f"Timesteps: {args.timesteps:,}")

    # Define supported algorithms
    SUPPORTED_ALGORITHMS = ['ppo', 'recurrent_ppo', 'ensemble']

    # Determine which algorithms to train
    if args.algorithms == 'all':
        algorithms = SUPPORTED_ALGORITHMS.copy()
    else:
        algorithms = [a.strip().lower() for a in args.algorithms.split(',')]
        invalid_algorithms = [a for a in algorithms if a not in SUPPORTED_ALGORITHMS]
        if invalid_algorithms:
            print(f"❌ ERROR: Invalid algorithm(s): {', '.join(invalid_algorithms)}")
            sys.exit(1)

    # Optimization: Ensemble includes PPO/RPPO
    if 'ensemble' in algorithms:
        if 'ppo' in algorithms or 'recurrent_ppo' in algorithms:
            print("\nℹ️  Optimization: 'ensemble' training includes PPO and RecurrentPPO.")
            algorithms = [a for a in algorithms if a not in ['ppo', 'recurrent_ppo']]

    print(f"Algorithms: {', '.join([a.upper() for a in algorithms])}")
    print()

    # Training loop
    for symbol in symbols:
        if len(symbols) > 1:
            print(f"\n{'='*80}")
            print(f"  PROCESSING SYMBOL: {symbol}")
            print(f"{ '='*80}\n")

        for i, agent_type in enumerate(algorithms, 1):
            print(f"\n📌 Step {i}/{len(algorithms)}: Training {agent_type.upper()} on {symbol}")
            try:
                # 1. Train
                save_dir = train_model(symbol, agent_type, args.timesteps)
                
                # 2. Generate Initial Backtest (for evaluation tools)
                print(f"\n📊 Generating initial backtest results...")
                validate_backtest.run_on_demand_backtest(save_dir, symbol, agent_type)
                print(f"   Backtest generated.")

            except Exception as e:
                logger.error(f"Skipping {agent_type} for {symbol} due to error: {e}")
                continue

    print("\n✨ All training tasks complete!")
    print("👉 Next steps:")
    print("   1. Run 'python eval_training.py' to compare performance and detect pathologies.")
    print("   2. Run 'python validate_backtest.py' for detailed mathematical validation.")
    print("=" * 80)


if __name__ == "__main__":
    main()