# Stock Analysis AI 📈

A comprehensive stock analysis system powered by **Ollama (Gemma3)**, **LangChain ReAct agents**, **LSTM ensemble neural networks**, and **Panel** for intelligent stock prediction and analysis with real-time progress tracking.

## Features

🤖 **Intelligent ReAct Agent**: LangChain ReAct agent with Gemma3 that plans and executes analysis tasks  
🧠 **LSTM Ensemble Models**: 3-model ensemble for robust 30-day stock price forecasting  
📈 **Interactive Visualizations**: Dynamic Plotly charts with historical data, predictions, and volume analysis  
💬 **Modern UI Interface**: Panel-based web application with real-time updates and responsive design  
⚡ **Real-Time Progress**: Live training progress with epoch-by-epoch loss tracking and visualization  
🎯 **Smart Query Processing**: Automatic stock symbol extraction and intelligent tool selection  
📊 **Comprehensive Analysis**: Complete metrics, trend analysis, and confidence scores  
🎛️ **Three-Column Layout**: Input controls, chat interface, and analysis results with interactive charts  

## Architecture

The system uses a **ReAct (Reasoning + Acting) agent** that intelligently plans and executes analysis tasks using specialized tools:

### Components

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

### Data Flow

```
User Input → Panel Interface → Symbol Extraction
                    ↓
    ReAct Agent (Gemma3) → Query Analysis → Tool Planning
                    ↓
         Dynamic Tool Execution with Real-Time Progress
                    ↓
    [Stock Fetcher] → [LSTM Ensemble] → [Visualizer]
         ↓               ↓                ↓
    Data Store    → Predictions    → Interactive Charts
                    ↓
         Agent Reasoning & Response Generation
                    ↓
    Chat Response + Real-Time Results Display + Loss Plots
```

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

### Start the Panel Application
```bash
source .venv/bin/activate  # Activate virtual environment
panel serve panel_app.py --port 5007 --show --autoreload
```

The app will open in your browser at `http://localhost:5007`

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

### Application Interface

**Left Sidebar - Input Controls:**
- **Chat Input**: Natural language queries with auto-scroll chat window
- **Quick Analysis**: Enter stock symbols for instant analysis with customizable periods (2y/5y)
- **Analysis Templates**: Pre-built analysis types (Basic, Trend, Risk Assessment)

**Middle Column - Conversation:**
- Real-time chat interface with AI assistant responses
- Progress updates showing analysis steps and completion status
- Clean, focused conversation flow with automatic scrolling

**Right Column - Analysis Results:**
- **Real-Time Progress**: Live training progress with loss/epoch visualization
- **Interactive Charts**: Price predictions, volume analysis, and trend charts
- **Metrics Dashboard**: Current price, predictions, trends with color-coded indicators
- **Expandable Results**: Organized display of comprehensive analysis data


## File Structure

```
stock_agent_ollama/
├── panel_app.py           # Panel web application with three-column layout
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
        "panel": {
            "port": 5007,
            "title": "Stock Analysis AI",
            "autoreload": True
        }
    }
```

## Key Dependencies

- **panel** (1.3.8): Modern web application framework with reactive components
- **langchain** (0.1.0): Agent orchestration and LLM integration
- **ollama** (0.1.7): Local LLM integration for Gemma3 model
- **yfinance** (0.2.58): Stock data fetching with rate limiting
- **tensorflow** (2.15.0): LSTM neural network training and ensemble modeling
- **plotly** (5.17.0): Interactive data visualization and real-time charts
- **pandas/numpy**: Data processing, analysis, and numerical computations

