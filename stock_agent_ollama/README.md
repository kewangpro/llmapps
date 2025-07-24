# Stock Analysis AI 📈

A comprehensive stock analysis system powered by **Ollama (Gemma3)**, **LangChain agents**, **LSTM neural networks**, and **Streamlit** for interactive chat-based stock prediction and analysis.

## Features

🤖 **AI-Powered Analysis**: Uses Ollama's Gemma3 model through LangChain agents  
📊 **LSTM Predictions**: Deep learning models for 30-day stock price forecasting  
📈 **Interactive Visualizations**: Dynamic charts showing trends and predictions  
💬 **Chat Interface**: Natural language interaction for stock queries  
🔧 **Modular Tools**: Separate tools for data fetching, prediction, and visualization  
🎯 **Smart Symbol Extraction**: Automatically identifies stock symbols from natural language  
📊 **Stock Comparison**: Side-by-side analysis of multiple stocks with separate visualizations  
📈 **Progress Tracking**: Real-time progress indicators during analysis  
🎛️ **Collapsible UI**: Organized interface with expandable AI analysis sections  

## Architecture

The system consists of three main LangChain tools orchestrated by an intelligent agent:

1. **Stock Fetcher Tool** (`stock_fetcher.py`)
   - Fetches historical stock data using yfinance with retry logic
   - Handles rate limiting and API errors gracefully
   - Retrieves company information and market metrics

2. **LSTM Predictor Tool** (`lstm_predictor.py`)
   - Trains LSTM neural networks on historical price data
   - Generates 30-day future price predictions with confidence metrics
   - Provides trend analysis and performance statistics
   - Optimized for M1/M2 Macs with legacy Adam optimizer support

3. **Visualizer Tool** (`visualizer.py`)
   - Creates interactive Plotly charts for price trends
   - Shows historical data vs predictions with volume analysis
   - Generates comprehensive trend analysis visualizations

4. **Stock Analysis Agent** (`stock_agent.py`)
   - LangChain ReAct agent with Ollama (Gemma3) integration
   - Orchestrates tools based on user queries with intelligent reasoning
   - Provides detailed analysis and investment insights
   - Includes comprehensive callback logging system

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
- **Quick Analysis**: Enter any stock symbol for instant comprehensive analysis
- **Analysis Templates**: Pre-built queries for common analysis types (Basic Analysis, Trend Analysis, Risk Assessment)

**Chat Interface:**
- Natural language processing with smart symbol extraction
- Context-aware conversations with memory
- Real-time progress tracking with step-by-step updates
- Collapsible AI analysis sections for organized display

**Comparison Mode:**
- Side-by-side stock analysis in separate tabs
- Complete metrics, charts, and AI analysis for each stock
- Automatic comparison summary with trend analysis

## System Architecture

```
User Query → Streamlit Interface → Stock Symbol Extraction
                    ↓
            Progress Tracking → LangChain Agent (Gemma3) → Tool Selection
                    ↓
    [Stock Fetcher] → [LSTM Predictor] → [Visualizer]
                    ↓
            AI Analysis & Insights Generation
                    ↓
        Response + Interactive Charts + Metrics
                    ↓
    Single Stock View OR Comparison Tabs View
```

## File Structure

```
data_model_ollama/
├── app.py                 # Streamlit chat interface with comparison support
├── stock_agent.py         # LangChain agent with comprehensive logging
├── stock_fetcher.py       # yfinance data fetching with rate limiting
├── lstm_predictor.py      # LSTM prediction optimized for M1/M2 Macs
├── visualizer.py          # Plotly visualization and chart generation
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── stock_analysis.log     # Application logs (auto-created)
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
            "sequence_length": 60,
            "prediction_days": 30,
            "epochs": 50,
            "batch_size": 32
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

## Troubleshooting

### Common Issues:

**1. "Ollama Not Connected" Error**
- Ensure Ollama is running: `ollama serve`
- Verify Gemma3 is installed: `ollama list`
- Check if port 11434 is available

**2. Stock Symbol Not Found**
- Use proper stock symbols (e.g., "AAPL" not "Apple")
- Check symbol spelling and market availability
- Try popular symbols: AAPL, GOOGL, MSFT, TSLA, AMZN

**3. yfinance Rate Limiting (429 Errors)**
- The system includes automatic retry logic with exponential backoff
- Wait a few minutes if rate limits persist
- System handles rate limiting automatically

**4. LSTM Training Errors**
- Ensure sufficient historical data (minimum 70 days)
- Check TensorFlow installation
- Verify memory availability for model training
- System includes M1/M2 Mac optimizer compatibility

**5. Import Errors**
- Activate virtual environment: `source .venv/bin/activate`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Performance Tips:

- **First-time Setup**: Initial Ollama model loading may take 1-2 minutes
- **LSTM Training**: Prediction generation takes 30-60 seconds per stock
- **Memory Usage**: Each analysis uses ~200-500MB RAM for model training
- **Rate Limiting**: System automatically handles Yahoo Finance API limits
- **Comparison Mode**: Analysis of multiple stocks runs sequentially with progress tracking

## Example Workflows

### Single Stock Analysis

1. **Start the system**:
   ```bash
   ollama serve              # Terminal 1
   source .venv/bin/activate # Terminal 2
   streamlit run app.py      # Terminal 2
   ```

2. **Ask a question**: "Evaluate the risk profile of investing in AAPL stock"

3. **System processes**:
   - Shows progress: "Fetching AAPL historical data..."
   - Extracts "AAPL" from natural language query
   - Shows progress: "Training LSTM model for AAPL..."
   - Trains LSTM model on price patterns (50 epochs)
   - Shows progress: "Creating AAPL charts and visualizations..."
   - Creates interactive price, volume, and trend charts
   - Shows progress: "Generating AI analysis for AAPL..."
   - Provides comprehensive AI analysis via Gemma3

4. **Review results**:
   - Current vs predicted price metrics
   - Interactive Plotly charts with zoom and hover features
   - Detailed AI analysis in expandable section

### Stock Comparison

1. **Ask**: "Compare GOOGL vs MSFT"

2. **System processes both stocks**:
   - Progress tracking for each stock separately
   - Fetches data, trains models, creates visualizations for both
   - Generates comparison summary

3. **Review results**:
   - Summary comparison in chat
   - Separate tabs for GOOGL and MSFT
   - Complete analysis (metrics, charts, AI insights) for each stock

## Advanced Features

**Progress Tracking:**
- Real-time step-by-step progress indicators
- Detailed status messages during analysis
- Progress bars showing completion percentage

**Smart Query Processing:**
- Context-aware stock symbol extraction from natural language
- Support for comparison queries ("X vs Y", "compare X and Y")
- Automatic fallback to general agent for non-analysis queries

**Comparison Analysis:**
- Side-by-side stock analysis in tabbed interface
- Complete visualization and analysis for each stock
- Comparison summary with trend analysis

**Robust Error Handling:**
- Graceful handling of API rate limits and network issues
- Fallback mechanisms for missing stock information
- User-friendly error messages with actionable suggestions

**UI Enhancements:**
- Collapsible AI analysis sections
- Organized metrics display
- Interactive chart integration

## Limitations & Disclaimers

⚠️ **Important Notes:**

- This is for **educational purposes only**
- **Not financial advice** - conduct your own research
- LSTM predictions are based on historical patterns and may not reflect future performance
- Market conditions, news, and external factors significantly impact stock prices
- Always consult with financial professionals before making investment decisions
- Stock data provided by Yahoo Finance - respect their terms of service

## Technical Details

- **AI Model**: Ollama Gemma3:latest with LangChain ReAct agent framework
- **Prediction Model**: 3-layer LSTM neural network with dropout regularization
- **Data Source**: Yahoo Finance API via yfinance library with rate limiting protection
- **Visualization**: Plotly for interactive charts with hover information and zoom capabilities
- **Framework**: LangChain for intelligent agent orchestration and tool management
- **Interface**: Streamlit with chat-style messaging and real-time updates
- **Memory Management**: Conversation buffer memory for context-aware interactions
- **Progress System**: Real-time callback-based progress tracking
- **Comparison Mode**: Tabbed interface for side-by-side stock analysis

## License

This project is for educational purposes. Stock data is provided by Yahoo Finance. Please respect their terms of service and use responsibly.