# Stock Analysis AI 📈

A comprehensive stock analysis system powered by **Ollama (Gemma3)**, **LangChain ReAct agents**, **LSTM ensemble neural networks**, and **Streamlit** for intelligent stock prediction and analysis with real-time progress tracking.

## Features

🤖 **Intelligent ReAct Agent**: LangChain ReAct agent with Gemma3 that plans and executes analysis tasks  
🧠 **LSTM Ensemble Models**: 3-model ensemble for robust 30-day stock price forecasting  
📈 **Interactive Visualizations**: Dynamic charts with historical data, predictions, and volume analysis  
💬 **Chat Interface**: Natural language interaction with real-time progress updates  
⚡ **Real-Time Progress**: Live training progress with epoch updates and loss tracking  
🎯 **Smart Query Processing**: Automatic stock symbol extraction and intelligent tool selection  
📊 **Comprehensive Analysis**: Complete metrics, trend analysis, and confidence scores  
🎛️ **Organized UI**: Expandable results section with metrics and interactive charts  

## Architecture

The system uses a **ReAct (Reasoning + Acting) agent** that intelligently plans and executes analysis tasks using specialized tools:

1. **ReAct Agent** (`stock_agent.py`)
   - LangChain ReAct agent powered by Ollama Gemma3
   - Analyzes user queries and plans appropriate tool sequence
   - Dynamic tool selection based on user requirements
   - Real-time progress callbacks and comprehensive logging

2. **Stock Fetcher Tool** (`stock_fetcher.py`)
   - Fetches historical stock data using yfinance
   - Intelligent data validation and preprocessing
   - Stores data in efficient data store architecture

3. **LSTM Predictor Tool** (`lstm_predictor.py`)
   - Trains 3-model LSTM ensemble for robust predictions
   - Real-time training progress with epoch and loss tracking
   - 30-day price forecasting with confidence metrics
   - Advanced technical indicators and feature engineering

4. **Visualizer Tool** (`visualizer.py`)
   - Creates interactive Plotly charts and visualizations
   - Historical price trends with prediction overlays
   - Volume analysis and trend indicators
   - Comprehensive market insights generation

## Prerequisites

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai/download
```

### 2. Install Gemma3 Model
```bash
ollama pull gemma3:latest
```

### 3. Start Ollama Service
```bash
ollama serve
# Keep this running in a separate terminal
```

## Installation

1. **Clone/Navigate to the project directory**
```bash
cd /path/to/data_model_ollama
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Start the Streamlit App
```bash
source .venv/bin/activate  # Activate virtual environment
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Example Queries

**Basic Analysis:**
- "Analyze AAPL stock"
- "What's the trend for Tesla stock?"
- "Give me insights on GOOGL performance"

**Predictions:**
- "Predict NVDA stock price for the next 30 days"
- "Show me MSFT price forecast with charts"
- "Forecast AMZN trends and predictions"

**Risk Assessment:**
- "Evaluate the risk profile of investing in AAPL stock"
- "What's the volatility analysis for META?"
- "Assess investment risks for TSLA"

**Comparisons:**
- "Compare AAPL vs GOOGL stocks"
- "Compare GOOGL vs MSFT"
- "Which is better investment: TSLA vs RIVN?"

### Quick Analysis Features

**Sidebar Tools:**
- **Quick Analysis**: Enter any stock symbol for instant comprehensive analysis with 2-year default period
- **Period Selection**: Choose between 2y (default) or 5y historical data
- **Analysis Templates**: Pre-built queries for common analysis types (Basic Analysis, Trend Analysis, Risk Assessment)

**Chat Interface:**
- Natural language processing with ReAct agent reasoning
- Dynamic tool selection based on query analysis
- Real-time progress tracking with detailed step updates
- LSTM training progress with epoch and loss monitoring

**Analysis Results:**
- Expandable results section with organized metrics
- Interactive charts with zoom, hover, and filtering
- Comprehensive trend analysis and confidence scores
- Data store architecture for efficient information sharing

## System Architecture

```
User Query → Streamlit Chat Interface → Symbol Extraction
                    ↓
    ReAct Agent (Gemma3) → Query Analysis → Tool Planning
                    ↓
         Dynamic Tool Execution with Progress Tracking
                    ↓
    [Stock Fetcher] → [LSTM Ensemble] → [Visualizer]
         ↓               ↓                ↓
    Data Store    → Predictions    → Charts & Insights
                    ↓
         Agent Reasoning & Response Generation
                    ↓
    Chat Response + Expandable Analysis Results
```

## File Structure

```
stock_agent_ollama/
├── app.py                 # Streamlit chat interface with real-time progress
├── stock_agent.py         # ReAct agent with dynamic tool orchestration
├── stock_fetcher.py       # yfinance data fetching with data store integration
├── lstm_predictor.py      # LSTM ensemble training with progress callbacks
├── visualizer.py          # Interactive Plotly charts and insights
├── data_store.py          # Efficient data sharing between tools
├── utils.py               # Utility functions and symbol extraction
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── logs/                  # Application logs directory
    └── stock_analysis.log # Main application log file (auto-created)
```

## Configuration

Modify settings in `config.py`:

```python
def get_config():
    return {
        "ollama": {
            "model": "gemma3:latest",
            "temperature": 0.1,
            "timeout": 30
        },
        "lstm": {
            "sequence_length": 120,
            "prediction_days": 30,
            "epochs": 75,
            "batch_size": 32,
            "ensemble_size": 3
        },
        "stock_data": {
            "default_period": "2y",
            "supported_periods": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        },
        "streamlit": {
            "page_title": "Stock Analysis AI",
            "page_icon": "📈",
            "layout": "wide"
        }
    }
```

## Key Dependencies

- **streamlit** (1.29.0): Web interface framework
- **langchain** (0.1.0): Agent orchestration and LLM integration
- **yfinance**: Stock data fetching with improved rate limiting
- **tensorflow** (2.15.0): LSTM neural network training
- **plotly**: Interactive data visualization
- **pandas/numpy**: Data processing and analysis

