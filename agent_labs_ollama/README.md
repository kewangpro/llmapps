# Agent Labs

AI-powered chat interface with intelligent multi-agent orchestration and real-time tool execution.

> 📸 **[View Screenshots & Demos](./docs/README.md#screenshots--demos)** | 📚 **[Full Documentation](./docs/)**

## Key Features

- **Multi-Agent Orchestration** - Intelligent task routing to specialized agents
- **Real-Time Tool Execution** - 10+ powerful tools for file analysis, web search, data processing, visualization
- **MCP Integration** - Connect to external Model Context Protocol servers for extended capabilities
- **Visual Content Analysis** - AI-powered image analysis with actual visual understanding
- **Interactive Results** - Charts, images, presentations displayed directly in chat
- **File Upload Support** - Automatic content analysis and tool selection
- **Multiple LLM Support** - Ollama, OpenAI, Google Gemini with vision capabilities
- **Downloadable Outputs** - Generated files saved with timestamps for easy access

## Architecture

**Frontend**: Next.js with WebSocket communication and real-time streaming
**Backend**: FastAPI with multi-agent orchestration system
**LLM**: Flexible provider support (Ollama, OpenAI, Gemini)
**MCP**: Model Context Protocol integration for external tool servers

**Flow**: User input → Orchestrator → Specialized agents → Tool execution → Real-time results

## Available Tools

### Built-in Tools
#### General (5)
- **File Search** - Intelligent file discovery and pattern matching
- **Web Search** - Real-time web search and information retrieval
- **System Info** - Comprehensive system monitoring and diagnostics
- **Presentation** - PowerPoint generation with downloadable outputs
- **Visualization** - Interactive chart generation from data files

#### Analytics (5)
- **Data Processing** - CSV/JSON conversion, text analysis, data transformation
- **Cost Analysis** - Financial data analysis with downloadable CSV reports and spending insights
- **Image Analysis** - Visual content analysis with metadata extraction
- **Stock Analysis** - Financial market data and technical analysis
- **Forecast** - LSTM neural network time series prediction and forecasting

### MCP Tools
- **External Integration** - Connect to MCP servers for additional specialized tools
- **Dynamic Discovery** - Automatically discover and integrate available MCP tools
- **Flexible Configuration** - Configure multiple MCP servers via environment variables

*Each tool has specialized agents with automatic parameter extraction and interactive result display.*


## Quick Start

### Docker (Recommended)
```bash
git clone <repository-url>
cd agent_labs_ollama
docker-compose up --build
```
**Access**: http://localhost:3000

### Manual Setup
1. **Clone and Install**:
   ```bash
   git clone <repository-url>
   cd agent_labs_ollama
   pip install -r requirements.txt
   npm install
   ```

2. **Setup Ollama**:
   ```bash
   # Install Ollama locally
   ollama serve
   ollama pull gemma3:latest
   ollama pull llama3.1:latest
   ```

3. **Run Application**:
   ```bash
   # Single command starts both frontend and backend
   python main.py
   ```
   **Access**: http://localhost:3000

**Environment Variables**: Create `.env` file with:
```bash
# LLM Configuration
LLM_PROVIDER=ollama  # or openai, gemini
LLM_MODEL=gemma3:latest

# Optional API Keys
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
GOOGLE_SEARCH_API_KEY=your_google_search_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# MCP Server Configuration (optional)
MCP_SERVERS=example_server
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
MCP_SERVERS=example_server,custom_server

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
"Hello world"  # Will use example_server:echo tool

# With multiple tools
"Analyze this data and echo the results"  # Uses both MCP and built-in tools
```

## Usage

1. **Select tools** from sidebar based on your task
2. **Upload files** or type questions
3. **View results** with interactive charts, images, and downloadable content

**Examples**: "Analyze this cost data", "Check system performance", "Create a chart from this CSV"

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
```

#### Environment Setup for Production
```bash
# Set environment variables in Cloud Run
gcloud run services update agent-labs \
  --set-env-vars="LLM_PROVIDER=ollama,LLM_MODEL=gemma3:latest" \
  --project=YOUR_PROJECT_ID
```

#### Monitoring
- **Build Logs**: Google Cloud Console → Cloud Build → History
- **Runtime Logs**: Google Cloud Console → Cloud Run → Service Logs
- **Health Check**: `GET /health` endpoint for service status

### Docker Deployment

```bash
# Build and run with Docker
docker build -t agent-labs .
docker run -p 3000:3000 -p 8000:8000 agent-labs
```

### Local Development with Production Configuration

```bash
# Use production environment variables
cp .env.example .env
# Edit .env with your configuration
npm run dev
```

## Development

**Tech Stack**: Next.js + FastAPI + Ollama
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
- ✅ **Multi-Agent System** - 11+ specialized agents with intelligent orchestration
- ✅ **Real-Time Streaming** - WebSocket-based communication with character-level streaming
- ✅ **Tool Integration** - 10+ powerful tools with automatic parameter extraction
- ✅ **MCP Protocol Support** - Connect to external MCP servers for extended functionality
- ✅ **File Management** - Timestamped outputs saved to dedicated outputs folder

**Capabilities**
- ✅ **Data Processing** - CSV/JSON conversion, text analysis, duplicate removal
- ✅ **Visual Analysis** - AI-powered image understanding and metadata extraction
- ✅ **Financial Analytics** - Cost analysis, stock market data visualization, and time series forecasting
- ✅ **Content Generation** - PowerPoint presentations and interactive charts
- ✅ **System Integration** - File search, web search, system monitoring
- ✅ **External Tool Integration** - Dynamic discovery and execution of MCP server tools

### Contributing
1. Fork repository
2. Create feature branch
3. Submit pull request

## License
MIT License