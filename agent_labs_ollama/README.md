# Agent Labs

AI-powered chat interface with intelligent multi-agent orchestration and real-time tool execution.

> 📸 **[View Screenshots & Demos](./docs/README.md#screenshots--demos)** | 📚 **[Full Documentation](./docs/)**

## Key Features

- **Multi-Agent Orchestration** - Intelligent task routing to specialized agents
- **Real-Time Tool Execution** - File analysis, web search, data visualization, cost analysis
- **Visual Content Analysis** - AI-powered image analysis with actual visual understanding
- **Interactive Results** - Charts, images, presentations displayed directly in chat
- **File Upload Support** - Automatic content analysis and tool selection
- **Multiple LLM Support** - Ollama, OpenAI, Google Gemini with vision capabilities

## Architecture

**Frontend**: Next.js with WebSocket communication
**Backend**: FastAPI with multi-agent orchestration
**LLM**: Ollama integration with model flexibility

**Flow**: User input → Orchestrator → Specialized agents → Tool execution → Real-time results

## Available Tools

**General**: File search, web search, system info, presentations
**Analytics**: Cost analysis, data visualization, code analysis, image analysis, stock analysis
**AI-Powered**: Visual content analysis with real image understanding, intelligent parameter extraction

*Each tool has specialized agents with automatic parameter extraction and interactive result display.*

### Specialized Agents

- **ImageAnalysisAgent** - Visual content analysis with AI-powered image understanding
- **PresentationAgent** - PowerPoint generation from content and files
- **FileSearchAgent** - Intelligent file discovery and content analysis
- **WebSearchAgent** - Real-time web search and information retrieval
- **CostAnalysisAgent** - Financial data analysis and visualization

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
- `GET /health` - Health check and system status
- `WS /ws/{client_id}` - Real-time chat with multi-agent orchestration
- `POST /api/llm/configure` - Configure LLM provider and model

### Recent Updates

**v2.0** (Latest)
- ✅ **Enhanced Image Analysis** - Real visual content analysis with AI understanding
- ✅ **Multi-LLM Support** - OpenAI GPT-4, Google Gemini, Ollama with vision capabilities
- ✅ **Improved Agent Architecture** - Better separation of concerns between tools and agents
- ✅ **Fixed JSON Serialization** - Proper handling of EXIF metadata and complex data types
- ✅ **Cloud Deployment** - Google Cloud Build and Cloud Run support

### Contributing
1. Fork repository
2. Create feature branch
3. Submit pull request

## License
MIT License