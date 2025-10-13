# Agent Labs

AI-powered chat interface with intelligent multi-agent orchestration and real-time tool execution.

> 📸 **[View Screenshots & Demos](./docs/README.md#screenshots--demos)** | 📚 **[Full Documentation](./docs/)**

## Key Features

- **Multi-Agent Orchestration** - Intelligent task routing with 14+ specialized agents
- **Real-Time Tool Execution** - 12 powerful built-in tools for file analysis, web search, data processing, visualization, presentations, flight search, and hotel search
- **Tool Mentions** - Use `@tool_name` syntax to explicitly select tools in your messages
- **MCP Integration** - Connect to external Model Context Protocol servers for extended capabilities
- **Visual Content Analysis** - AI-powered image analysis with actual visual understanding
- **Interactive Results** - Charts, images, presentations, and flight information displayed directly in chat
- **File Upload Support** - Automatic content analysis and smart tool selection
- **Multiple LLM Support** - Ollama, OpenAI, Google Gemini with vision capabilities
- **Downloadable Outputs** - Generated files saved with timestamps for easy access
- **Cloud Deployment Ready** - Production-ready deployment to Google Cloud Run

## Architecture

**Frontend**: Next.js with WebSocket communication and real-time streaming
**Backend**: FastAPI with multi-agent orchestration system
**LLM**: Flexible provider support (Ollama, OpenAI, Gemini)
**MCP**: Model Context Protocol integration for external tool servers

**Flow**: User input → Orchestrator → Specialized agents → Tool execution → Real-time results

## Available Tools

### Built-in Tools (12 Total)
#### General Tools (7)
- **File Search** - Intelligent file discovery and pattern matching across filesystems
- **Web Search** - Real-time web search with current information retrieval
- **System Info** - Comprehensive system monitoring, CPU, memory, disk, and network diagnostics
- **Presentation** - PowerPoint generation from structured data with downloadable PPTX files
- **Flight Search** - Search for flights between cities
- **Hotel Search** - Search for hotel accommodations
- **Visualization** - Interactive chart and graph generation from CSV/JSON data

#### Analytics Tools (5)
- **Data Processing** - CSV/JSON conversion, text analysis, data transformation and cleaning
- **Cost Analysis** - Financial data analysis with business unit breakdown and spending insights
- **Image Analysis** - Visual content analysis with metadata extraction and object detection
- **Stock Analysis** - Financial market data analysis with Yahoo Finance integration
- **Forecast** - LSTM neural network time series prediction with visualization support

### MCP Tools
- **External Integration** - Connect to MCP servers for additional specialized tools
- **Dynamic Discovery** - Automatically discover and integrate available MCP tools
- **Flexible Configuration** - Configure multiple MCP servers via environment variables

*Each tool has specialized agents with automatic parameter extraction and interactive result display.*


## Quick Start

### Docker Compose (Recommended for Local Development with Ollama)
```bash
git clone <repository-url>
cd agent_labs_ollama
docker-compose up --build
```
**Access**: http://localhost:3000
**Note**: This setup includes both the application and an Ollama service

### Manual Setup
1. **Clone and Install**:
   ```bash
   git clone <repository-url>
   cd agent_labs_ollama
   pip install -r requirements.txt
   npm install
   ```

2. **Setup Ollama** (if using Ollama provider):
   ```bash
   # Install and start Ollama
   ollama serve
   # Pull models
   ollama pull llama3.2:latest
   ollama pull llama3.1:latest
   ```

3. **Run Application**:
   ```bash
   # Single command starts both frontend and backend
   python main.py
   ```
   **Access**: http://localhost:3000

**Environment Variables**: Create `.env` file (see [.env.example](.env.example)):
```bash
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key       # For OpenAI models
GEMINI_API_KEY=your_gemini_key       # For Google Gemini models
OLLAMA_BASE_URL=http://localhost:11434  # For Ollama (local or remote)

# Google Search API (optional, for web search tool)
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# MCP Server Configuration (optional)
MCP_SERVERS=example-server
MCP_EXAMPLE_SERVER_URL=http://localhost:8000
MCP_EXAMPLE_SERVER_TOOLS=echo,get_time,search_web
MCP_EXAMPLE_SERVER_DESCRIPTION=Basic MCP server for testing
```

## MCP Integration

### Overview
Agent Labs supports the Model Context Protocol (MCP) for connecting to external tool servers. MCP tools appear in their own category and are dynamically discovered.

### Configuration
Configure MCP servers in your `.env` file:
```bash
# Define server names (comma-separated)
MCP_SERVERS=example-server,custom-server

# Configure each server
MCP_EXAMPLE_SERVER_URL=http://localhost:8000
MCP_EXAMPLE_SERVER_TOOLS=echo,get_time
MCP_EXAMPLE_SERVER_DESCRIPTION=Example MCP server

MCP_CUSTOM_SERVER_URL=http://localhost:8001
MCP_CUSTOM_SERVER_TOOLS=search,analyze
MCP_CUSTOM_SERVER_DESCRIPTION=Custom analytics server
```

### Tool Selection
- MCP tools appear in the "MCP" category in the sidebar
- Tool names follow the pattern: `Server-Name:Tool-Name`
- Quick Tips are automatically generated for all MCP tools
- Select MCP tools alongside built-in tools for hybrid workflows

### Usage Examples
```bash
# With echo tool selected
"Hello world"  # Will use example-server:echo tool

# With multiple tools
"Analyze this data and echo the results"  # Uses both MCP and built-in tools
```

## Usage

### Tool Selection Methods

**1. Sidebar Selection**
- Click tools in the sidebar to select them
- Selected tools will be used for your query
- Multiple tools can be selected for complex workflows

**2. @ Mention (Explicit Selection)**
- Type `@` to see available tools
- Select from dropdown or type `@tool_name` directly
- When using `@`, ONLY mentioned tools will be used (sidebar selection is ignored)
- Examples:
  - `@flight_search find flights from Seattle to Tokyo`
  - `@web_search latest AI developments`
  - `@data_processing @visualization analyze and chart this data`

### Examples

**Travel Search**:
- `@flight_search find flights from San Francisco to Tokyo on December 20`
- `@hotel_search find hotels in Paris from January 10 to January 15`

**Information & Analysis**:
- `@web_search latest AI developments`
- `@system_info check system performance and show metrics`
- `@file_search search for files containing 'config'`

**Data Processing**:
- `@data_processing @visualization create charts from this CSV data`
- `@cost_analysis analyze cost per business unit and create presentation`
- `@stock_analysis get Tesla stock analysis with forecast`

## Deployment

### Google Cloud Platform

The application can be deployed to Google Cloud using Cloud Build and Cloud Run:

#### Prerequisites
- Google Cloud SDK installed and configured
- Google Cloud project with billing enabled
- Cloud Build and Cloud Run APIs enabled

#### Deploy to Google Cloud
```bash
# From project root directory
~/google-cloud-sdk/bin/gcloud builds submit . --project=YOUR_PROJECT_ID

# Example for existing deployment
~/google-cloud-sdk/bin/gcloud builds submit . --project=agent-labs-1758473505
```

#### Environment Setup for Production
```bash
# Set environment variables in Cloud Run
~/google-cloud-sdk/bin/gcloud run services update agent-labs \
  --set-env-vars="LLM_PROVIDER=openai,OPENAI_API_KEY=your_key" \
  --region=us-central1 \
  --project=YOUR_PROJECT_ID
```

#### Monitoring
- **Build Logs**: Google Cloud Console → Cloud Build → History
- **Runtime Logs**: Google Cloud Console → Cloud Run → Service Logs
- **Health Check**: `GET /health` endpoint for service status

### Docker Deployment (Single Container)

```bash
# Build and run single container (for cloud deployment)
docker build -t agent-labs .
docker run -p 3000:3000 \
  -e OPENAI_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  agent-labs
```
**Note**: This is a multi-stage build that bundles both frontend and backend. Best for cloud deployment (Cloud Run, etc.).

### Local Development

```bash
# Use environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
python main.py
```
**Access**: http://localhost:3000

## Development

**Tech Stack**: Next.js 14 + FastAPI + Multiple LLM Providers (Ollama/OpenAI/Gemini)
**Communication**: WebSocket for real-time interaction
**Architecture**: Multi-agent pattern with specialized tool execution

### API Endpoints
- `GET /api/tools` - Available tools and agent information
- `GET /api/models` - Available LLM models (Ollama, OpenAI, Gemini)
- `POST /api/select-model` - Change active LLM model and provider
- `GET /health` - Health check and system status
- `WS /ws/{client_id}` - Real-time chat with multi-agent orchestration

### Current Features

**Architecture**
- ✅ **Multi-Agent System** - 14+ specialized agents with intelligent orchestration and precise tool selection
- ✅ **Real-Time Streaming** - WebSocket-based communication with character-level streaming
- ✅ **Tool Integration** - 12 powerful built-in tools with automatic parameter extraction and smart routing
- ✅ **Tool Mentions** - `@tool_name` syntax for explicit tool selection in messages
- ✅ **MCP Protocol Support** - Connect to external MCP servers for extended functionality
- ✅ **File Management** - Timestamped outputs saved to dedicated outputs folder
- ✅ **Cloud Deployment** - Production-ready deployment to Google Cloud Run with health monitoring

**Capabilities**
- ✅ **Data Processing** - CSV/JSON conversion, text analysis, data transformation and cleaning
- ✅ **Visual Analysis** - AI-powered image understanding with metadata extraction and object detection
- ✅ **Financial Analytics** - Cost analysis, stock market data, business unit breakdowns, and LSTM forecasting
- ✅ **Content Generation** - PowerPoint presentations, interactive charts, and downloadable reports
- ✅ **System Integration** - File search, web search, system monitoring with comprehensive diagnostics
- ✅ **Travel Search** - Flight and hotel search with pricing, schedules, ratings, and booking information
- ✅ **External Tool Integration** - Dynamic discovery and execution of MCP server tools

### Contributing
1. Fork repository
2. Create feature branch
3. Submit pull request

## License
MIT License
