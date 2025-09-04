# 🌍 AI Trip Planner

Smart multi-city trip planning with dual collaboration modes. Plan complete itineraries with flights, hotels, and activities using either fast LLM reasoning or comprehensive Google Search analysis.

## ✨ Features

- **🚀 Dual Modes**: Simple Mode (fast LLM reasoning, 30-60s) + Comprehensive Mode (Google Search analysis, 3-5min)
- **📱 Modern Web App**: Flutter-based responsive interface  
- **🖥️ CLI Tool**: Command-line interface for direct planning
- **🔄 Flexible Configuration**: Simple Mode works offline, Comprehensive Mode uses Google Search API
- **🎯 Intelligent Curation**: Smart flight and hotel selection with primary recommendations and alternatives
- **🏨 Optimized Selection**: Algorithmic scoring based on price, rating, and convenience factors

## 🏗️ How It Works

**Simple Mode (Fast):**
```
User Request → FastAPI Server → TravelAgent → Ollama Model → LLM Reasoning → Trip Plan (30-60s)
```

**Comprehensive Mode (Detailed):**
```
User Request → FastAPI Server → 5 Specialized Agents → Google Search API → Synthesis Agent → Trip Plan (3-5min)
```

**Stack:**
- **Frontend**: Flutter Web App + CLI Tool
- **Backend**: FastAPI with LangChain ReAct framework
- **AI**: Dual-mode system - single TravelAgent (Simple) + 5 specialized agents (Comprehensive)
- **Data**: Pure LLM reasoning (Simple) + Google Search API (Comprehensive)

## 🚀 Quick Start

### 1. Setup Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama and pull model (any Ollama model works)
ollama serve
ollama pull llama3.2:latest  # or gemma2, gemma3, qwen2.5, etc.
# Configure model in config.py: OLLAMA_MODEL=gemma3:latest

# Start API server
python main.py
```

### 2. Run Web App
```bash
cd mobile_app
./run.sh  # Automatically sets up and runs in Chrome
```

### 3. Try CLI Tool
```bash
# Plan a trip via command line (Simple Mode - default, fast)
python run.py --origin Seattle --destinations Tokyo --start-date 2025-09-02 --duration 7

# Multi-city with Comprehensive Mode (detailed analysis with Google Search)
python run.py --origin NYC --destinations Paris London --start-date 2025-06-01 --duration 10 --mode comprehensive
```

### 4. Optional: Google Search API (for Comprehensive Mode)
Copy `.env.example` to `.env` and add your Google Search credentials to enable Comprehensive Mode with real-time data. Simple Mode works perfectly without this.

## 📱 What You Get

**Complete Trip Plans:**
- ✈️ Flight recommendations with airlines, times, and prices
- 🎯 Curated flight options (outbound/return separation with alternatives)
- 🏨 Best hotel per city (auto-selected by rating + price optimization)
- 🔄 Primary recommendations + alternative options for flexible planning
- 📅 Daily itineraries with activities and local tips
- 💰 Budget estimates and cost breakdowns

**Smart Interface:**
- 🎯 Clean, mobile-optimized web interface
- 🤖 Real-time LLM reasoning with structured output
- 📱 Responsive design for all devices
- ⚡ One-command setup and development

## 🔧 API Usage

**Main Endpoint:** `POST /plan-trip`
```json
{
  "origin": "San Francisco",
  "destinations": ["Tokyo", "Seoul"],
  "start_date": "2025-12-01", 
  "duration_days": 8,
  "budget": "medium",
  "preferences": "food, culture"
}
```

**Curation Endpoints:**
- `POST /curate-flights` - Manual flight curation with scoring
- `POST /curate-hotels` - Manual hotel curation and selection
- `GET /curation-status` - Check curation system status

**Other Endpoints:** Health checks, system status, and interactive docs at `/docs`

## 🤖 LLM Agent Architecture

**Dual Collaboration Modes** using LangChain's ReAct framework:

**Simple Mode (Fast Production):**
- Single TravelAgent with LLM reasoning tools
- 30-60 second execution time for optimal performance
- Pure LLM knowledge without external API dependencies
- Works offline - no configuration required

**Comprehensive Mode (Detailed Analysis):**
- 5 specialized agents using Google Search tools + synthesis agent
- Sequential collaboration: Budget → Flight → Accommodation → Activity → Synthesis
- 3-5 minute execution time with real-time market data
- Requires Google Search API credentials for enhanced accuracy

**Core Features:**
- **Chain-of-Thought**: Multi-step planning with reasoning traces
- **Tool Selection**: Autonomous decision-making for flight/hotel/activity searches  
- **Structured Output**: Standardized format parsing for reliable data extraction
- **Mode-Specific Tools**: LLM reasoning (Simple) vs Google Search (Comprehensive)
- **Clean Architecture**: No tool contamination between modes

**Available Tools:**
- **Simple Mode**: LLM-based Flight/Hotel/Activity Search, Budget Analysis, Route Optimization
- **Comprehensive Mode**: Google Search tools + MasterSynthesisAgent for result combination

## 📁 Structure

```
├── main.py              # FastAPI server + trip planning API
├── run.py               # CLI tool for trip planning
├── models.py            # Pydantic data models & validation
├── config.py            # Configuration management
├── curation.py          # Intelligent flight & hotel selection system
├── agents/              # LangChain ReAct agent system
│   ├── travel_agent.py                 # Simple Mode single agent (LLM reasoning)
│   ├── master_synthesis_agent.py       # Comprehensive Mode synthesis agent
│   ├── langchain_multi_agent_system.py # Dual-mode coordination system
│   ├── flight_planning_agent.py        # Specialized flight search agent
│   ├── accommodation_agent.py          # Hotel search specialist
│   ├── activity_agent.py               # Activity recommendation agent
│   ├── budget_planning_agent.py        # Budget analysis agent
│   ├── travel_tools.py                 # LLM reasoning tools (Simple Mode)
│   ├── google_enhanced_tools.py        # Google Search tools (Comprehensive Mode)
│   └── langchain_base_agent.py         # ReAct framework base
├── schemas/             # Agent output validation schemas
│   └── agent_output_schema.py          # Standardized output format validation
├── mobile_app/          # Flutter web interface
└── services/            # Google Search & utilities
```

## 🧪 Testing

Use the CLI tool to test the system:
```bash
python run.py --origin Seattle --destinations Tokyo --start-date 2025-09-02 --duration 7
```

Or check the API directly:
```bash
curl http://localhost:8000/health
```

## 🚀 Production

**Backend:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Web App:**
```bash
cd mobile_app && flutter build web
```

## 🔑 Google Search API (Required for Comprehensive Mode)

To enable Comprehensive Mode with real-time data, add Google Search credentials to `.env`:
1. Get API key from [Google Developers Console](https://developers.google.com/custom-search/v1/overview)
2. Create Custom Search Engine at [cse.google.com](https://cse.google.com)
3. Add `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` to `.env`

**Note**: Simple Mode works perfectly without any API credentials using pure LLM reasoning.

## 🔬 Technical Details

**LLM Agent Framework:**
- **ReAct Pattern**: Reasoning + Acting with tool calling
- **Dual Architecture**: Single agent (Simple) + Multi-agent (Comprehensive) modes
- **Format Alignment**: Structured output parsing with exact pattern matching
- **Error Handling**: Robust parsing with graceful degradation
- **Temperature**: Low temperature (0.1) for consistent format compliance
- **Model Support**: Works with any Ollama model (gemma3, llama3.2, qwen2.5, etc.)

**Data Processing:**
- **Mode-Specific Processing**: LLM reasoning (Simple) vs Google Search API (Comprehensive)
- **Structured Extraction**: Regex parsing of standardized LLM output
- **Multi-City Support**: Complex routing with proper date sequencing
- **Clean Architecture**: No tool contamination between collaboration modes
- **Smart Curation**: Algorithmic flight and hotel selection with primary/alternative structure

**Curation System:**
- **Flight Scoring**: Price (70%) + Duration (30%) with outbound/return separation
- **Hotel Scoring**: Rating (60%) + Price (40%) optimization
- **Flexible Output**: Primary recommendations + alternatives for user choice
- **Dual Format Support**: Handles both curated and legacy data structures

---

**Built with LangChain ReAct + Ollama (Multi-model Support) + Flutter**