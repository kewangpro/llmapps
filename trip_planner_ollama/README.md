# AI Trip Planner with LangChain Agents

A production-ready AI-powered trip planning system featuring LangChain agent framework with reasoning capabilities, Google Search tools integration, and intelligent travel planning. Built with LangChain AgentExecutor using Ollama, featuring a modern Flutter mobile app.

## 🎯 Key Features

### 🤖 LangChain Agent Framework
- **🧠 AgentExecutor with ReAct**: Reasoning and Acting agents with chain-of-thought planning
- **🔗 Multi-Agent Collaboration**: 5 specialized agents working together
- **🛠️ Google Search Tools**: Real-time data through Google Search API integration
- **🔄 Automatic Tool Selection**: Agents autonomously choose the best tools for each task

### 🚀 Intelligent Agent System
- **✈️ Flight Planning Agent**: Smart flight search and route optimization
- **🏨 Accommodation Agent**: Hotel and accommodation research with preferences
- **🎯 Activity Agent**: Local activities and experiences matching interests  
- **💰 Budget Planning Agent**: Financial analysis and cost optimization
- **🗺️ Master Travel Agent**: Comprehensive trip coordination and planning

### 🔍 Google Search Integration
- **🌐 Web Search Tools**: Real-time activity and attraction searches
- **✈️ Flight Search**: Google-powered flight data and pricing
- **🏨 Hotel Search**: Comprehensive accommodation searches
- **🔄 Intelligent Fallbacks**: Works with or without API keys using agent reasoning

### 📱 Modern Mobile Experience
- **🌐 Flutter Web App**: Cross-platform native performance in browser
- **📊 Data Source Indicators**: Visual badges showing LangChain agent sources
- **📱 Responsive Design**: Optimized for all device sizes
- **🤖 Agent Insights**: Shows reasoning chains and collaboration details

## 🏗️ Architecture

```
┌─────────────────┐    HTTP API     ┌─────────────────┐   Agent Tasks   ┌─────────────────┐
│ Flutter Web App │ ───────────────▶│  FastAPI Server │ ───────────────▶│ LangChain Agents│
│  (Mobile UI)    │ ◀─────────────── │ (Orchestration) │ ◀─────────────── │ (5 Specialists) │
└─────────────────┘    JSON         └─────────────────┘   Results       └─────────────────┘
                                                                                 │
                                                                                 │
                                                                     ┌───────────┴───────────┐
                                                                     │                       │
                                                                     ▼                       ▼
                                                           ┌─────────────────┐   ┌─────────────────┐
                                                           │ Google Search   │   │   Ollama LLM    │
                                                           │ Tools & APIs    │   │ (mistral:latest)│
                                                           │ (if configured) │   │   (Reasoning)   │
                                                           └─────────────────┘   └─────────────────┘
                                                                     │                       │
                                                                     │           ┌───────────┘
                                                                     │           │
                                                                     ▼           ▼
                                                           ┌─────────────────────────────────┐
                                                           │    Agent Memory & Context       │
                                                           │    (Reasoning Chain Storage)    │
                                                           └─────────────────────────────────┘
```

**Architecture Flow:**
1. **Flutter Web App** → Sends trip requests to FastAPI server
2. **FastAPI Server** → Only orchestrates LangChain agents (no direct API calls)
3. **LangChain Agents** → Use both Ollama LLM (for reasoning) and Google Search tools (for data)
4. **Ollama LLM** → Provides reasoning, planning, and language capabilities to agents
5. **Google Search Tools** → Provide real-time data when agents decide to use them
6. **Agent Memory** → Stores reasoning chains and context from both LLM and tool interactions

**Components:**
- **Frontend**: Flutter Web App - Modern responsive mobile experience  
- **Backend**: FastAPI (Python) - Pure agent orchestration (no direct API usage)
- **Agent System**: 5 specialized LangChain agents that coordinate reasoning and tool usage
- **Reasoning Engine**: Ollama with mistral:latest model (configurable) for agent intelligence
- **Data Tools**: Google Search APIs (optional) for real-time travel information
- **Memory System**: Agent context and reasoning chain storage

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- Flutter SDK 3.10.0+ (for mobile app)
- Ollama with Mistral model (or other supported models)
- Optional: Google Search API key for enhanced real-time data

### 1. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (if not running)
ollama serve

# Pull Mistral model (default) or other supported models
ollama pull mistral:latest

# Alternative models you can use:
# ollama pull gemma3:latest
# ollama pull llama3.1:latest
# ollama pull phi3:latest

# Start the API server
python main.py
```

The API runs on `http://localhost:8000` with interactive docs at `/docs`

### 2. Mobile App Setup

```bash
# Navigate to mobile app directory
cd mobile_app

# Install Flutter dependencies
flutter pub get

# Run on web (recommended)
./run.sh
```

### 3. Google Search API Configuration (Optional)

For enhanced real-time data and custom configuration, copy `.env.example` to `.env`:

```bash
# Copy example configuration
cp .env.example .env

# Edit .env file and customize settings:
# Google Search API credentials (optional)
# GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
# GOOGLE_SEARCH_ENGINE_ID=your_custom_search_engine_id_here

# Model configuration
# OLLAMA_MODEL=mistral:latest  # Default, or use gemma3:latest, llama3.1:latest, etc.
# OLLAMA_TEMPERATURE=0.3
# AGENT_TIMEOUT=30.0

# Other optional settings
# CACHE_ENABLED=true
# DEBUG=false
# LOG_LEVEL=INFO
```

**Get Google Search API credentials:**
1. Visit [Google Developers Console](https://developers.google.com/custom-search/v1/overview)
2. Create a project and enable Custom Search API
3. Create a Custom Search Engine at [cse.google.com](https://cse.google.com)
4. Add your API key and Search Engine ID to `.env`

## 📱 Mobile App Features

### Modern Flutter Experience
- **🤖 LangChain Agent Indicators**: Visual badges showing agent sources:
  - 🤖 **LangChain** - Indigo badge for LangChain agent reasoning
  - ☁️ **API** - Blue badge for direct API calls  
  - ⚡ **Fallback** - Orange badge for fallback data
- **🧠 Agent Insights**: Shows reasoning steps and agent collaboration details
- **🔄 Unified Timeline**: Chronological display merging flights and daily plans
- **📱 Responsive Design**: Optimized for mobile, tablet, and desktop

### Trip Planning Interface
- **🗺️ Multi-City Input**: Plan complex round trips with multiple destinations
- **📅 Smart Validation**: Real-time form validation and user guidance
- **✈️ Enhanced Flight Cards**: Airlines, times, prices with agent reasoning
- **📋 Daily Plan Cards**: Detailed activities, accommodations, transportation
- **🤖 Agent Details**: See which agents contributed to each recommendation

## 🔧 API Endpoints

### Core Planning Endpoint

#### `POST /plan-trip`
LangChain agent-powered trip planning with reasoning capabilities.

```json
{
  "origin": "San Francisco",
  "destinations": ["Tokyo", "Seoul"],
  "start_date": "2024-12-01", 
  "duration_days": 8,
  "budget": "medium",
  "preferences": "food, culture, modern attractions"
}
```

**Response includes:**
- Comprehensive trip plan with flights, hotels, and daily activities
- Agent reasoning chains and collaboration details
- Tool usage statistics and confidence scores
- Personalized recommendations based on preferences

### System Endpoints

#### `GET /health`
Detailed health check with system information and status.

#### `GET /`
Basic health check endpoint with system status.

#### `GET /api-status`
LangChain agent system status and capabilities.

#### `GET /test-ollama`
Test Ollama connectivity and available models.

#### `GET /test-flight-cards`
Test endpoint for mobile app integration verification.

#### `GET /debug-fallback`
Debug fallback data creation for development.

#### `GET /debug-text-extraction`
Debug text extraction with sample agent output.

#### `POST /reset-error-stats`
Reset error statistics (production maintenance).

#### `GET /docs`
Interactive API documentation (Swagger UI).

## 🤖 LangChain Agent System

### Specialized Agents

#### 🗺️ MasterTravelAgent
- **Role**: Comprehensive trip coordination and planning
- **Tools**: Flight search, hotel search, activity search, budget analysis, trip synthesis
- **Capabilities**: Multi-destination routing, preference matching, overall coordination

#### ✈️ FlightPlanningAgent  
- **Role**: Flight search and route optimization
- **Tools**: Flight search, route optimization
- **Capabilities**: Best flight options, connection analysis, pricing optimization

#### 🏨 AccommodationAgent
- **Role**: Hotel and accommodation research
- **Tools**: Hotel search with preference filtering
- **Capabilities**: Location-based recommendations, amenity matching, budget alignment

#### 🎯 ActivityAgent
- **Role**: Local activities and experiences
- **Tools**: Activity and attraction search
- **Capabilities**: Interest-based recommendations, cultural experiences, local insights

#### 💰 BudgetPlanningAgent
- **Role**: Financial analysis and optimization
- **Tools**: Budget analysis and allocation
- **Capabilities**: Cost estimation, budget distribution, financial optimization

### Agent Collaboration
- **Reasoning Chains**: Each agent maintains detailed reasoning about decisions
- **Tool Sharing**: Agents can use each other's tools when needed
- **Memory Management**: Conversation buffer maintains context across interactions
- **Error Handling**: Intelligent fallbacks when agent execution fails

## 📁 Project Structure

```
trip_planner_ollama/
├── README.md                           # This documentation
├── requirements.txt                    # Python dependencies
├── .env.example                        # Configuration template
├── config.py                           # Configuration management
├── main.py                             # Main FastAPI server with LangChain agents
├── models.py                           # Enhanced data models
├── test_agent.py                       # Agent system testing
├── test_api.py                         # API integration testing
├── test_data_extraction.py             # Data extraction testing
├── agents/                             # LangChain agent system
│   ├── __init__.py                     # Agent system exports
│   ├── README.md                       # Agent documentation
│   ├── langchain_base_agent.py         # Base LangChain agent class
│   ├── langchain_multi_agent_system.py # Multi-agent coordination
│   ├── master_travel_agent.py          # Master coordination agent
│   ├── flight_planning_agent.py        # Flight planning specialist
│   ├── accommodation_agent.py          # Hotel accommodation specialist
│   ├── activity_agent.py               # Activities and experiences
│   ├── budget_planning_agent.py        # Budget optimization specialist
│   └── travel_tools.py                 # LangChain travel tools
├── services/                           # Core services
│   ├── __init__.py                     # Service exports
│   ├── google_travel_search.py         # Google Search integration
│   └── error_handler.py                # Production error handling
└── mobile_app/                         # Flutter web application
    ├── lib/                            # Dart source code
    │   ├── main.dart                   # Flutter main app
    │   ├── models/                     # Dart data models
    │   ├── screens/                    # UI screens
    │   └── services/                   # API services
    ├── web/                            # Web assets and icons
    ├── pubspec.yaml                    # Flutter dependencies
    ├── analysis_options.yaml           # Flutter analysis config
    ├── README.md                       # Mobile app documentation
    └── run.sh                          # Quick start script
```

## 🧪 Testing

### Comprehensive Test Suite

The project includes three specialized test tools for different aspects of the system:

#### 1. Data Extraction Testing (`test_data_extraction.py`)
Tests the core data extraction and parsing functionality that converts agent outputs into structured trip plans.

```bash
# Run data extraction tests
python test_data_extraction.py
```

**What it tests:**
- ✅ Flight and hotel regex pattern extraction
- ✅ Agent output parsing (both old and new formats)  
- ✅ TripPlan model creation and validation
- ✅ Response transformation for frontend
- ✅ Mock agent result processing

#### 2. Agent System Testing (`test_agent.py`)
Comprehensive testing of the LangChain agent system with recent improvements and multi-agent collaboration.

```bash
# Run agent system tests
python test_agent.py
```

**What it tests:**
- 🤖 MasterTravelAgent functionality and completion
- ✈️ Single city trip planning (Seattle → Tokyo)
- 🌏 Multi-city trip planning (Seattle → Tokyo → Shanghai)
- 🔧 Tool usage validation (flight_search, hotel_search, activity_search)
- 📊 Agent status and reasoning validation
- ⚡ Simple query processing speed
- 🛠️ Recent bug fixes and improvements validation

**Sample output:**
```
🧪 TEST 1: Single City Trip (Seattle → Tokyo)
✅ Agent completion fixed (no timeout)
✅ Flight search tool working  
✅ Hotel search tool working
✅ Activity search tool working (error fixed)
```

#### 3. API Integration Testing (`test_api.py`)
End-to-end HTTP API testing with realistic scenarios, error handling, and performance monitoring.

```bash
# Start the API server first
python main.py

# In another terminal, run API tests
python test_api.py
```

**What it tests:**
- 🔍 Health check endpoint (`/health`)
- 🚀 Trip planning endpoint (`/plan-trip`) with various scenarios:
  - Simple Tokyo trip (basic functionality)
  - Multi-city Asian tour (complex routing)
  - Budget European trip (budget constraints)
  - Invalid request handling (error validation)
- ⏱️ Response time monitoring
- 🛡️ Error handling and timeout behavior
- 📊 HTTP status code validation
- 📋 Response data structure validation

**Sample output:**
```
📋 TEST 2: Multi-City Asian Tour
Status: success
Response Time: 45.23s
✅ Test PASSED
   Route: san francisco → tokyo → seoul
   Flights: 3, Hotels: 2, Daily Plans: 5
```

### Quick Tests

```bash
# Test backend functionality
source .venv/bin/activate
python -c "import main; print('✅ Backend imports successfully')"

# Test mobile app
cd mobile_app && flutter analyze
```

### Manual API Testing

```bash
# Test basic planning
curl -X POST http://localhost:8000/plan-trip \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "Seattle",
    "destinations": ["Tokyo"],
    "start_date": "2024-12-01",
    "duration_days": 5,
    "budget": "medium",
    "preferences": "food, culture"
  }'

# Test system status
curl http://localhost:8000/api-status

# Test Ollama integration
curl http://localhost:8000/test-ollama
```

### Test Execution Strategy

**For Development:**
```bash
# Quick validation during development
python test_data_extraction.py

# Full agent validation after changes
python test_agent.py
```

**For Production Deployment:**
```bash
# Start server
python main.py &

# Full API validation
python test_api.py

# Validate all systems
python test_data_extraction.py && python test_agent.py
```

**Test Coverage:**
- 🔧 **Core functionality**: Data extraction and parsing
- 🤖 **Agent intelligence**: LangChain agent system behavior  
- 🌐 **API integration**: HTTP endpoints and error handling
- 📱 **Mobile compatibility**: Response format validation
- 🚀 **Performance**: Response time and timeout monitoring

## 🌐 Production Deployment

### Backend Deployment

```bash
# Production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker deployment
docker build -t trip-planner-langchain .
docker run -p 8000:8000 --env-file .env trip-planner-langchain
```

### Mobile App Deployment  

```bash
cd mobile_app

# Build for web deployment
flutter build web

# Serve static files (example with nginx)
cp -r build/web/* /var/www/html/
```

## 🌟 What Makes This Special?

### 🤖 True AI Agent Framework
- **Reasoning Capabilities**: Agents think through problems step-by-step
- **Autonomous Decision Making**: Agents choose tools and strategies independently  
- **Chain-of-Thought Planning**: Visible reasoning process for transparency
- **Multi-Agent Collaboration**: Specialized agents work together on complex tasks

### 🚀 Production-Ready
- **Clean Architecture**: FastAPI server only orchestrates agents (no direct API calls)
- **Agent Autonomy**: Agents independently decide when to use Google Search tools  
- **Intelligent Fallbacks**: Works perfectly without any API credentials
- **Robust Error Handling**: Comprehensive error recovery and fallback systems
- **Separation of Concerns**: Server → Agents → Tools → APIs (clean layers)

### 📊 Enhanced User Experience
- **Agent Transparency**: See exactly which agents worked on each recommendation
- **Reasoning Visibility**: Understand how decisions were made
- **Mobile-First Design**: Optimized for modern mobile browsing
- **Real-Time Intelligence**: Live agent reasoning with tool integration

## 🔑 Getting Google Search API Keys (Optional)

The system works perfectly without API keys using intelligent agent reasoning. For enhanced real-time data:

### Google Search API Setup
1. Visit [Google Developers Console](https://developers.google.com/custom-search/v1/overview)
2. Create a project and enable Custom Search API
3. Get your API key from the credentials section
4. Create a Custom Search Engine at [cse.google.com](https://cse.google.com)
5. Configure your search engine to search the entire web
6. Get your Search Engine ID
7. Add both to your `.env` file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`) 
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Built with ❤️ using LangChain Agent Framework for next-generation AI travel planning**