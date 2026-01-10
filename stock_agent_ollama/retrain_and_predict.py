#!/usr/bin/env python3
"""
LSTM Retraining & Prediction Tool

1. Retrains the LSTM ensemble model for a specific stock symbol.
2. Generates a fresh prediction using the new model.
3. Validates the prediction (checks for gaps).

Usage: 
    python retrain_and_predict.py --symbol AAPL
    python retrain_and_predict.py --symbol MSFT --horizon 30 --epochs 100
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.tools.stock_fetcher import StockFetcher

from src.tools.lstm_predictor import LSTMPredictor

from src.tools.portfolio_manager import portfolio_manager

from src.config import Config



# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



# Color codes

GREEN = '\033[92m'

RED = '\033[91m'

YELLOW = '\033[93m'

BLUE = '\033[94m'

RESET = '\033[0m'

BOLD = '\033[1m'



def process_symbol(symbol: str, horizon: int, epochs: int, batch_size: int, predictor: LSTMPredictor, fetcher: StockFetcher):

    """Retrain and predict for a single symbol"""

    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")

    print(f"{BOLD}{BLUE}PROCESSING: {symbol} (Horizon={horizon}){RESET}")

    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

    

    # 1. Fetch Data

    print(f"{BLUE}1. Fetching 2 years of historical data for {symbol}...{RESET}")

    end_date = datetime.now()

    start_date = end_date - timedelta(days=730) 

    

    try:

        data = fetcher.fetch_stock_data(

            symbol=symbol,

            start_date=start_date.strftime('%Y-%m-%d'),

            end_date=end_date.strftime('%Y-%m-%d'),

            interval="1d"

        )

        

        if data.empty:

            print(f"{RED}Error: No data found for {symbol}{RESET}")

            return False

            

        current_price = data['Close'].iloc[-1]

        print(f"  Fetched {len(data)} records. Last date: {data.index[-1].date()}")

        print(f"  Current Price: ${current_price:.2f}")

        

    except Exception as e:

        print(f"{RED}Error fetching data for {symbol}: {e}{RESET}")

        return False



    # 2. Train Model

    print(f"\n{BLUE}2. Starting training process...{RESET}")

    print(f"  Epochs: {epochs}")

    print(f"  Batch Size: {batch_size}")

    

    def progress_callback(data):

        if data.get('type') == 'lstm_training_progress':

            epoch = data.get('epoch')

            loss = data.get('loss')

            val_loss = data.get('val_loss')

            print(f"  Model {data.get('model')}/{data.get('total_models')} - Epoch {epoch}: Loss {loss:.4f} | Val Loss {val_loss:.4f}", end='\r')

        elif data.get('type') == 'lstm_training_complete':

            print(f"\n  Model {data.get('model')} complete.")

            

    try:

        metrics = predictor.train_ensemble(

            data=data,

            symbol=symbol,

            validation_split=0.2,

            epochs=epochs,

            batch_size=batch_size,

            progress_callback=progress_callback,

            horizon=horizon

        )

        

        print(f"\n{GREEN}TRAINING COMPLETE for {symbol}{RESET}")

        print(f"  RMSE: {metrics['rmse']:.4f}")

        print(f"  MAE:  {metrics['mae']:.4f}")

        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")

        

        # 3. Generate Predictions

        print(f"\n{BLUE}3. Generating prediction...{RESET}")

        

        prediction_days = max(30, horizon)

        

        result = predictor.predict(

            symbol=symbol,

            data=data,

            days=prediction_days

        )

        

        predictions = result['predictions']

        dates = result['dates']

        conf_lower = result['confidence_lower']

        conf_upper = result['confidence_upper']

        

        print(f"\n{BOLD}Prediction Results:{RESET}")

        print(f"{'Date':<12} {'Predicted':<10} {'Lower':<10} {'Upper':<10} {'Change %':<10}")

        print("-" * 55)

        

        first_pred = predictions[0]

        gap_pct = ((first_pred - current_price) / current_price) * 100

        

        for i in range(min(5, len(predictions))): # Show first 5 days for brevity in batch

            date_str = dates[i]

            pred = predictions[i]

            lower = conf_lower[i]

            upper = conf_upper[i]

            prev_val = current_price if i == 0 else predictions[i-1]

            change = ((pred - prev_val) / prev_val) * 100

            print(f"{date_str:<12} ${pred:<9.2f} ${lower:<9.2f} ${upper:<9.2f} {change:+.2f}%")

            

        print("...")

        

        final_pred = predictions[-1]

        total_change = ((final_pred - current_price) / current_price) * 100

        

        print("-" * 55)

        print(f"Current Price:   ${current_price:.2f}")

        print(f"First Predicted: ${first_pred:.2f} (Gap: {gap_pct:+.2f}%)")

        print(f"Final Predicted: ${final_pred:.2f} (Total Change: {total_change:+.2f}%)")

        

        if abs(gap_pct) > 5.0:

            print(f"\n{YELLOW}⚠️  WARNING: Large gap detected!{RESET}")

        else:

            print(f"\n{GREEN}✅ Gap check passed{RESET}")

            

        return True

            

    except Exception as e:

        print(f"\n{RED}Processing {symbol} failed: {e}{RESET}")

        return False



def main():

    parser = argparse.ArgumentParser(description='Retrain and Predict with LSTM')

    

    symbol_group = parser.add_mutually_exclusive_group(required=True)

    symbol_group.add_argument('--symbol', type=str, help='Stock symbol(s) to train - single or comma-separated')

    symbol_group.add_argument('--watchlist', action='store_true', help='Train all symbols from default watchlist')

    

    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon in days (default: 1)')

    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help=f'Number of training epochs (default: {Config.EPOCHS})')

    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE, help=f'Batch size (default: {Config.BATCH_SIZE})')

    

    args = parser.parse_args()

    

    # Determine symbols

    if args.watchlist:

        symbols = portfolio_manager.load_portfolio("default")

        if not symbols:

            print(f"{RED}Error: Watchlist is empty or not found{RESET}")

            sys.exit(1)

    else:

        symbols = [s.strip().upper() for s in args.symbol.split(',')]



    print(f"\n{BOLD}{BLUE}{'#'*60}{RESET}")

    print(f"{BOLD}{BLUE}LSTM BATCH RETRAINING & PREDICTION{RESET}")

    print(f"{BOLD}{BLUE}Symbols: {', '.join(symbols)}{RESET}")

    print(f"{BOLD}{BLUE}{'#'*60}{RESET}\n")



    predictor = LSTMPredictor()

    fetcher = StockFetcher()

    

    success_count = 0

    for symbol in symbols:

        if process_symbol(symbol, args.horizon, args.epochs, args.batch_size, predictor, fetcher):

            success_count += 1

            

    print(f"\n{BOLD}{GREEN if success_count == len(symbols) else YELLOW}{'='*60}{RESET}")

    print(f"{BOLD}BATCH COMPLETED: {success_count}/{len(symbols)} successful{RESET}")

    print(f"{BOLD}{GREEN if success_count == len(symbols) else YELLOW}{'='*60}{RESET}\n")

if __name__ == "__main__":
    main()