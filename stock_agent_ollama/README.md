# Stock Analysis AI Platform

An AI-powered platform for stock analysis that lets you ask natural language questions like "Analyze AAPL" or "Predict Tesla's price" and get comprehensive analysis with charts and forecasts.

![Platform Demo](https://img.shields.io/badge/Status-Ready-success)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue)

## What It Does

- **Ask Questions in Plain English**: "Show me Apple's performance" or "What will Google stock be worth?"
- **AI Price Predictions**: 30-day forecasts using machine learning models
- **Technical Analysis**: RSI, MACD, moving averages, and other indicators
- **Interactive Charts**: Live data visualization with multiple chart types
- **Stock Comparisons**: Side-by-side analysis of different stocks

## Getting Started

### Requirements
- Python 3.9+
- 8GB RAM (16GB recommended)
- Internet connection

### Installation

1. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   python src/main.py
   ```

4. **Open your browser** to `http://localhost:5006`

## How to Use

Just type natural language questions in the web interface:

- **"Analyze AAPL"** - Get comprehensive stock analysis with charts
- **"Predict Google stock price"** - See 30-day AI forecasts
- **"Compare Apple vs Microsoft"** - Side-by-side performance charts
- **"What's Tesla's current price?"** - Real-time market data

### Supported Stocks
Popular stocks (AAPL, GOOGL, MSFT, AMZN, TSLA), indices (SPY, QQQ), crypto (BTC-USD, ETH-USD), and many others.

## Technical Details

### How It Works
- **Natural Language Processing**: Understands your questions and converts them to analysis tasks
- **Data Collection**: Fetches real-time stock data from Yahoo Finance
- **AI Predictions**: Uses LSTM neural networks trained on 60 days of historical data
- **Technical Analysis**: Calculates 17+ indicators including RSI, MACD, and Bollinger Bands
- **Visualization**: Creates interactive charts with Plotly

### Built With
- Python 3.9+ and TensorFlow for AI models
- Panel framework for the web interface
- Yahoo Finance API for stock data
- Plotly for interactive charts

## Testing

Verify everything is working:
```bash
python verify_setup.py
```

This runs a comprehensive test suite to ensure all components are functioning properly.

## Troubleshooting

**Import Errors**: Make sure the virtual environment is activated (`source .venv/bin/activate`) and dependencies are installed.

**Memory Issues**: Close other applications during AI model training (uses 2-4GB RAM).

**Network Issues**: Ensure internet connection is working for stock data fetching.

**TensorFlow Warnings**: LibreSSL and retracing warnings are normal on macOS and don't affect functionality.

## Project Structure

```
stock_agent_ollama/
├── src/                    # Source code
│   ├── agents/            # Natural language processing
│   ├── tools/             # Stock data, AI predictions, analysis
│   ├── ui/                # Web interface
│   └── main.py            # Start here
├── data/                  # Cached data and AI models
├── requirements.txt       # Dependencies
└── verify_setup.py        # Test script
```

## Disclaimer

This platform is for educational and research purposes only. The AI predictions and analysis are not financial advice. Always consult with financial professionals before making investment decisions.

## License

Educational and research use only. Ensure compliance with data provider terms when using financial data.