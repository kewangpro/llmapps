# 📈 Stock Analysis AI Platform

A powerful, AI-driven stock analysis platform that combines natural language processing, LSTM-based price predictions, and comprehensive technical analysis in an intuitive web interface.

![Platform Demo](https://img.shields.io/badge/Status-Ready-success)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![Panel](https://img.shields.io/badge/Panel-1.4.5-green)

## 🎯 Features

### 🔍 **Intelligent Stock Analysis**
- **Natural Language Queries**: Ask questions like "Analyze AAPL for the last 6 months"
- **Technical Analysis**: Comprehensive indicators including RSI, MACD, Bollinger Bands, and moving averages
- **Real-time Data**: Live stock prices and market data from Yahoo Finance
- **Smart Caching**: Efficient data caching for improved performance

### 🤖 **AI-Powered Predictions**
- **LSTM Neural Networks**: Deep learning models for price forecasting
- **Ensemble Models**: Multiple model combination for improved accuracy
- **Confidence Intervals**: Prediction uncertainty quantification
- **30-Day Forecasts**: Short-term price movement predictions

### 📊 **Interactive Visualizations**
- **Dynamic Charts**: Interactive Plotly-based stock charts
- **Multiple Chart Types**: Candlestick, line charts, and volume analysis
- **Technical Overlays**: Moving averages and technical indicators
- **Prediction Visualization**: Future price forecasts with confidence bands

### 💬 **Conversational Interface**
- **Natural Language Processing**: Understand complex stock analysis requests
- **Query Examples**: 
  - "Show me Tesla's performance"
  - "Predict Google stock price"
  - "Compare Apple vs Microsoft"
  - "What's the current price of Amazon?"

## 🚀 Quick Start

### Prerequisites
- **macOS 10.15+** (Catalina or later)
- **Python 3.9+** (currently tested with Python 3.9.6)
- **8GB RAM** (16GB recommended for LSTM training)
- **Internet connection** for stock data
- **Virtual environment** (`.venv` already configured)

### Installation

1. **Clone and Navigate**
   ```bash
   cd /Users/kewang/PyProjects/stock_agent_ollama
   ```

2. **Activate Virtual Environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python verify_setup.py
   ```

5. **Start the Application**
   ```bash
   python src/main.py
   ```

6. **Open Your Browser**
   Navigate to: `http://localhost:5006`

## 🎮 Usage Examples

### Basic Stock Analysis
```
Query: "Analyze AAPL"
Result: Comprehensive analysis with current price, technical indicators, and charts
```

### Price Predictions
```
Query: "Predict GOOGL for next month"
Result: 30-day LSTM forecast with confidence intervals
```

### Stock Comparison
```
Query: "Compare AAPL vs MSFT"
Result: Side-by-side performance comparison charts
```

### Current Price Information
```
Query: "What's Tesla's current price?"
Result: Real-time price, volume, and market cap data
```

## 🏗️ Architecture

### Core Components

```
📦 Stock Analysis AI Platform
├── 🎯 Query Processor (Natural Language Understanding)
├── 📊 Stock Data Fetcher (Yahoo Finance Integration)
├── 🤖 LSTM Predictor (Deep Learning Models)
├── 📈 Technical Analysis Engine (Indicators & Signals)
├── 🎨 Visualizer (Interactive Charts)
├── 💾 Cache Manager (Local Data Storage)
└── 🖥️ Panel UI (Web Interface)
```

### Technology Stack

- **Frontend**: Panel (Python web framework)
- **Backend**: Python 3.9+ with asyncio
- **Data Source**: Yahoo Finance API
- **Machine Learning**: TensorFlow/Keras LSTM models
- **Visualization**: Plotly interactive charts
- **Caching**: Local file-based cache system
- **UI Framework**: Panel with Bootstrap template

## 📋 Supported Query Types

### Analysis Queries
- `"Analyze [SYMBOL]"`
- `"Show me [SYMBOL] for the last [TIME_PERIOD]"`
- `"[SYMBOL] analysis"`
- `"Look at [SYMBOL]"`

### Prediction Queries
- `"Predict [SYMBOL]"`
- `"Forecast [SYMBOL]"`
- `"What will [SYMBOL] be worth?"`
- `"Future price of [SYMBOL]"`

### Comparison Queries
- `"Compare [SYMBOL1] vs [SYMBOL2]"`
- `"[SYMBOL1] and [SYMBOL2] comparison"`

### Price Queries
- `"[SYMBOL] current price"`
- `"How much is [SYMBOL]?"`
- `"Price of [SYMBOL]"`

### Supported Symbols
- **Popular Stocks**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, NFLX, TEAM, CRM, UBER, LYFT
- **Indices**: SPY, QQQ, IWM, VTI, DIA
- **Crypto**: BTC-USD, ETH-USD
- **And many more traditional stocks and ETFs...**

## 🔧 Configuration

### Environment Variables
```bash
# Panel Configuration
export PANEL_PORT=5006
export PANEL_HOST=localhost

# Cache Settings
export CACHE_TTL_SECONDS=3600

# Logging
export LOG_LEVEL=INFO
```

### Model Configuration
- **LSTM Sequence Length**: 60 days
- **Ensemble Size**: 3 models
- **Prediction Period**: 30 days
- **Training Epochs**: 50 (configurable)

## 📊 Performance Metrics

### Response Time Targets
- **Query Processing**: < 3 seconds
- **Chart Rendering**: < 500ms
- **LSTM Prediction**: < 2 minutes
- **Data Fetching**: < 1 second (cached)

### Accuracy Metrics
- **LSTM Models**: Directional accuracy typically 50-70% (varies by symbol and market conditions)
- **Technical Signals**: Combined indicator confidence scoring
- **Data Quality**: Real-time validation and cleaning
- **Model Training**: 3-model ensemble for improved robustness

## 🧪 Testing

### Run All Tests
```bash
python verify_setup.py
```

### Component Testing
```bash
# Test individual components
python -c "from src.tools.stock_fetcher import StockFetcher; print('Stock Fetcher OK')"
python -c "from src.tools.lstm_predictor import LSTMPredictor; print('LSTM Predictor OK')"
python -c "from src.agents.query_processor import QueryProcessor; print('Query Processor OK')"
```

## 🚨 Troubleshooting

### Common Issues

1. **TensorFlow Warnings**
   - LibreSSL warnings are normal on macOS and don't affect functionality
   - `tf.function retracing` warnings are normal during model training

2. **Memory Issues**
   - Close other applications when training LSTM models
   - Training uses approximately 2-4GB RAM per model

3. **Network Issues**
   - Check internet connection for Yahoo Finance data
   - Verify firewall settings allow Python network access
   - Cached data will be used when available to reduce API calls

4. **Import Errors**
   - Ensure virtual environment is activated with `source .venv/bin/activate`
   - Run `pip install -r requirements.txt` again if needed

### Performance Optimization
- **Cache Warming**: Let the system build cache for frequently used symbols
- **Model Training**: Train models during off-peak hours
- **Memory Management**: Monitor RAM usage during heavy operations

## 📁 Project Structure

```
stock_agent_ollama/
├── src/                          # Source code
│   ├── agents/                   # AI agents and processors
│   │   └── query_processor.py    # Natural language query handling
│   ├── tools/                    # Analysis and prediction tools
│   │   ├── stock_fetcher.py      # Yahoo Finance integration
│   │   ├── lstm_predictor.py     # LSTM neural network models
│   │   ├── technical_analysis.py # Technical indicators
│   │   └── visualizer.py         # Chart generation
│   ├── ui/                       # User interface components
│   │   └── components.py         # Panel UI components
│   ├── utils/                    # Utilities
│   │   └── cache_utils.py        # Caching system
│   ├── config.py                 # Configuration management
│   └── main.py                   # Application entry point
├── data/                         # Data storage
│   ├── cache/                    # Cached stock data
│   ├── models/                   # Trained LSTM models
│   └── logs/                     # Application logs
├── requirements.txt              # Python dependencies
├── verify_setup.py               # Setup verification
└── README.md                     # This file
```

## 🔮 Future Enhancements

### Near-term Features
- **More Technical Indicators**: Stochastic, Williams %R, ATR
- **Portfolio Analysis**: Multi-stock portfolio tracking
- **Export Capabilities**: PDF reports, CSV data export
- **Mobile Optimization**: Better responsive design

### Advanced Features
- **Sentiment Analysis**: News and social media sentiment
- **Options Analysis**: Options pricing and Greeks
- **Risk Modeling**: VaR, beta analysis
- **Real-time Alerts**: Price and indicator notifications

## ⚠️ Disclaimer

This platform is designed for educational and research purposes. The AI predictions and analysis should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

- **Not Financial Advice**: All predictions and analyses are for educational purposes only
- **Past Performance**: Historical data does not guarantee future results
- **Risk Warning**: Stock investments carry inherent risks
- **Data Accuracy**: While we strive for accuracy, data may contain errors

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using financial data.

## 🤝 Contributing

This is a local MacBook prototype focused on demonstrating core AI-powered stock analysis capabilities. The modular architecture makes it easy to extend and enhance functionality.

---

**Built with ❤️ for intelligent stock analysis**

*Ready to revolutionize your stock analysis workflow? Start exploring with natural language queries today!*