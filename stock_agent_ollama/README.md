# 🚀 AI Stock Analysis Platform

An intelligent financial analysis platform that combines **LSTM neural networks**, **AI-powered explanations**, and **comprehensive technical analysis** through an intuitive web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![AI](https://img.shields.io/badge/AI-Ollama%20%2B%20LSTM-orange)
![Educational](https://img.shields.io/badge/Purpose-Educational-purple)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 🎯 What It Does

- **🤖 AI Query Processing**: Natural language questions powered by Ollama with gemma3:latest model
- **🧠 LSTM Predictions**: 30-day stock price forecasts using ensemble neural networks
- **📊 Technical Analysis**: 17+ indicators including RSI, MACD, Bollinger Bands, and trading signals
- **💬 Educational Explanations**: AI-generated interpretations of technical concepts
- **📈 Interactive Charts**: Real-time visualization with Plotly and Panel dashboard

## 🚀 Quick Start

### 1. Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Ollama (Optional but Recommended)
```bash
# Install Ollama from https://ollama.ai
ollama pull gemma3:latest
```

### 3. Launch Platform
```bash
python src/main.py
# Open http://localhost:5006
```

## 💬 Usage Examples

**Natural Language Queries:**
```
"Analyze Apple stock performance"
"Predict Tesla price for next 30 days"  
"Compare Microsoft vs Google"
"Explain what RSI means for beginners"
```

**Quick Actions:** Click buttons for AAPL, GOOGL, MSFT, TSLA, AMZN analysis

## 🏗️ Architecture

### Data Flow
```
User Query → AI Processing (Ollama) → Stock Analysis → LSTM Prediction → Interactive Dashboard
     ↓              ↓ (fallback)           ↓                ↓                    ↓
Natural Language → Regex Patterns → Technical Indicators → Charts & Explanations
```

### Technology Stack
- **AI**: Ollama (gemma3:latest) with regex fallback
- **ML**: TensorFlow LSTM ensemble models
- **Data**: Yahoo Finance API with intelligent caching  
- **UI**: Panel framework with Plotly visualizations
- **Backend**: Python with asyncio and aiohttp

## ⚙️ Configuration

### Environment Variables
```bash
# Ollama Settings
OLLAMA_MODEL=gemma3:latest
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_ENABLED=true

# Panel Settings  
PANEL_PORT=5006
PANEL_HOST=localhost
```

### Health Checks
```bash
# Test core functionality
source .venv/bin/activate
python -c "from src.agents.query_processor import QueryProcessor; print('✅ Platform ready')"

# Test Ollama integration
python -c "
import asyncio
from src.agents.ollama_enhancer import OllamaEnhancer

async def test():
    enhancer = OllamaEnhancer()
    healthy = await enhancer.health_check()
    print(f'Ollama: {\"✅ Connected\" if healthy else \"⚠️ Using fallback\"}')
    await enhancer.close()

asyncio.run(test())
"
```

## 🎓 Educational Features

### Learning Applications
- **Technical Analysis**: Understand indicators like RSI, MACD, moving averages
- **Machine Learning**: Explore LSTM neural networks for time series prediction
- **Financial Concepts**: AI explanations adapted to your knowledge level
- **Market Analysis**: Learn to interpret trends, signals, and risk factors

### Pre-trained Models
Ready-to-use LSTM models available for: **AAPL**, **MSFT**, **AMZN**, **TEAM**

## 📁 Project Structure

```
stock_agent_ollama/
├── src/
│   ├── main.py                     # Application entry point
│   ├── agents/                     # AI & NLP processing
│   │   ├── query_processor.py      # Main query handler
│   │   ├── hybrid_query_processor.py  # Ollama integration
│   │   └── ollama_enhancer.py      # AI explanations
│   ├── tools/                      # Analysis & prediction
│   │   ├── lstm/                   # Neural network components
│   │   ├── stock_fetcher.py        # Data acquisition
│   │   ├── technical_analysis.py   # Indicators & signals
│   │   └── conversation_manager.py # Session management
│   └── ui/components.py            # Panel dashboard
├── data/                           # Runtime data
│   ├── models/lstm/                # Trained neural networks
│   ├── cache/                      # Stock data cache
│   └── conversations/              # Chat sessions
└── requirements.txt                # Dependencies
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **Import errors** | Activate virtual environment: `source .venv/bin/activate` |
| **Ollama unavailable** | Platform works with regex fallback, install Ollama for AI features |
| **Port 5006 in use** | Set `PANEL_PORT=5007` environment variable |
| **Memory issues** | Close other applications, LSTM training needs 4GB+ RAM |
| **No predictions** | LSTM models train automatically on first prediction request |

### Common Commands
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test Panel interface
python -c "import panel as pn; print('Panel version:', pn.__version__)"

# View application logs
tail -f data/logs/app.log
```

## 🎯 Key Features

### Hybrid Intelligence
- **AI-First**: Ollama processes complex natural language queries
- **Reliable Fallback**: Regex patterns ensure consistent functionality
- **Educational**: AI explanations make complex concepts accessible

### Advanced Analytics
- **Ensemble LSTM**: Multiple models for robust 30-day predictions
- **17+ Indicators**: Professional-grade technical analysis
- **Trading Signals**: BUY/SELL/HOLD recommendations with confidence
- **Risk Assessment**: Automated trend analysis and factor identification

### User Experience
- **Interactive Dashboard**: Real-time analysis with visual feedback
- **Conversation Memory**: Multi-turn educational discussions
- **Quick Actions**: One-click analysis for popular stocks
- **Responsive Design**: Works on desktop and tablet devices

## ⚖️ Educational License & Disclaimer

### Purpose
This platform is designed for **educational and research purposes**. It demonstrates:
- Modern AI techniques in financial analysis
- LSTM neural networks for time series forecasting  
- Natural language processing for domain applications
- Professional software development practices

### Important Disclaimers
- 📚 **Educational Use Only** - Not for actual trading decisions
- ⚠️ **No Financial Advice** - AI predictions are for learning purposes
- 🔬 **Research Tool** - Designed for academic and educational exploration
- 📊 **Past Performance** - Does not guarantee future results

### Risk Acknowledgment
- AI predictions are experimental and educational
- Markets are unpredictable and can change rapidly
- Always consult qualified financial professionals
- Use paper trading to test strategies safely

---

**Built for Financial AI Education** 💙

Perfect for students, researchers, and developers exploring AI applications in finance. Combines cutting-edge machine learning with educational best practices.