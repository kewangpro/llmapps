# 🌍 AI Trip Planner

Smart multi-city trip planning powered by pure LLM reasoning. Plan complete itineraries with flights, hotels, and activities using local AI models with autonomous decision-making.

## ✨ Features

- **🤖 Pure LLM Reasoning**: Advanced ReAct framework with autonomous tool usage
- **📱 Modern Web App**: Flutter-based responsive interface  
- **🖥️ CLI Tool**: Command-line interface for direct planning
- **🔄 Works Offline**: No API keys required (Google Search optional)
- **🎯 Intelligent Curation**: Smart flight and hotel selection with primary recommendations and alternatives
- **🏨 Optimized Selection**: Algorithmic scoring based on price, rating, and convenience factors

## 🏗️ How It Works

```
User Request → FastAPI Server → LLM Agent → Ollama (Gemma3) → Complete Trip Plan
```

**Stack:**
- **Frontend**: Flutter Web App + CLI Tool
- **Backend**: FastAPI with LangChain ReAct agent
- **AI**: Master travel agent with dual collaboration modes using Ollama/Gemma3
- **Data**: Google Search (optional) + intelligent fallbacks

## 🚀 Quick Start

### 1. Setup Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama and pull model
ollama serve
ollama pull gemma3:latest

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
# Plan a trip via command line
python run.py --origin Seattle --destinations Tokyo --start-date 2025-09-02 --duration 7

# Multi-city example
python run.py --origin NYC --destinations Paris London --start-date 2025-06-01 --duration 10
```

### 4. Optional: Google Search API
Copy `.env.example` to `.env` and add your Google Search credentials for enhanced real-time data (system works perfectly without this).

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

**Simple Mode (Recommended):**
- Single master agent with all 5 travel tools
- 25-second execution time for optimal performance
- Direct tool access with autonomous reasoning

**Comprehensive Mode:**
- 5 specialized agents in sequential collaboration
- Detailed multi-agent reasoning across budget, flight, accommodation, activity, and synthesis phases
- Extended execution time with enhanced analysis depth

**Core Features:**
- **Chain-of-Thought**: Multi-step planning with reasoning traces
- **Tool Selection**: Autonomous decision-making for flight/hotel/activity searches  
- **Structured Output**: Standardized format parsing for reliable data extraction
- **Pure LLM**: No fallback systems - full autonomous operation

**Available Tools**: Flight Search, Hotel Search, Activity Search, Budget Analysis, Route Optimization, Intelligent Curation

## 📁 Structure

```
├── main.py              # FastAPI server + trip planning API
├── run.py               # CLI tool for trip planning
├── models.py            # Pydantic data models & validation
├── config.py            # Configuration management
├── curation.py          # Intelligent flight & hotel selection system
├── agents/              # LangChain ReAct agent system
│   ├── master_travel_agent.py         # Main LLM reasoning agent
│   ├── langchain_multi_agent_system.py # Dual-mode coordination system
│   ├── flight_planning_agent.py       # Specialized flight search agent
│   ├── accommodation_agent.py          # Hotel search specialist
│   ├── activity_agent.py               # Activity recommendation agent
│   ├── budget_planning_agent.py        # Budget analysis agent
│   ├── travel_tools.py                 # Travel planning tools
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

## 🔑 Google Search API (Optional)

For enhanced real-time data, add Google Search credentials to `.env`:
1. Get API key from [Google Developers Console](https://developers.google.com/custom-search/v1/overview)
2. Create Custom Search Engine at [cse.google.com](https://cse.google.com)
3. Add `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` to `.env`

## 🔬 Technical Details

**LLM Agent Framework:**
- **ReAct Pattern**: Reasoning + Acting with tool calling
- **Format Alignment**: Structured output parsing with exact pattern matching
- **Error Handling**: Robust parsing with graceful degradation
- **Temperature**: Low temperature (0.1) for consistent format compliance

**Data Processing:**
- **Structured Extraction**: Regex parsing of standardized LLM output
- **Multi-City Support**: Complex routing with proper date sequencing
- **Real-time Integration**: Google Search API with intelligent fallbacks
- **Smart Curation**: Algorithmic flight and hotel selection with primary/alternative structure

**Curation System:**
- **Flight Scoring**: Price (70%) + Duration (30%) with outbound/return separation
- **Hotel Scoring**: Rating (60%) + Price (40%) optimization
- **Flexible Output**: Primary recommendations + alternatives for user choice
- **Dual Format Support**: Handles both curated and legacy data structures

---

**Built with LangChain ReAct + Ollama (Gemma3) + Flutter**