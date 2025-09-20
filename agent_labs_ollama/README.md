# Agent Labs

AI-powered chat interface with intelligent multi-agent orchestration and real-time tool execution.

> 📸 **[View Screenshots & Demos](./docs/README.md#screenshots--demos)** | 📚 **[Full Documentation](./docs/)**

## Key Features

- **Multi-Agent Orchestration** - Intelligent task routing to specialized agents
- **Real-Time Tool Execution** - File analysis, web search, data visualization, cost analysis
- **Interactive Results** - Charts, images, presentations displayed directly in chat
- **File Upload Support** - Automatic content analysis and tool selection
- **Multiple LLM Support** - Works with any Ollama-compatible model

## Architecture

**Frontend**: Next.js with WebSocket communication
**Backend**: FastAPI with multi-agent orchestration
**LLM**: Ollama integration with model flexibility

**Flow**: User input → Orchestrator → Specialized agents → Tool execution → Real-time results

## Available Tools

**General**: File search, web search, system info, presentations
**Analytics**: Cost analysis, data visualization, code analysis, image analysis, stock analysis

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
1. **Backend**: Python 3.9+, install requirements, run `python main.py`
2. **Frontend**: Node.js 18+, `npm install`, `npm run dev`
3. **Ollama**: Install locally, pull models (`gemma2`, `llama3.1`), run `ollama serve`

**Optional**: Configure Google Search API in `.env` for web search

## Usage

1. **Select tools** from sidebar based on your task
2. **Upload files** or type questions
3. **View results** with interactive charts, images, and downloadable content

**Examples**: "Analyze this cost data", "Check system performance", "Create a chart from this CSV"

## Development

**Tech Stack**: Next.js + FastAPI + Ollama
**Communication**: WebSocket for real-time interaction
**Architecture**: Multi-agent pattern with specialized tool execution

### API Endpoints
- `GET /api/tools` - Available tools
- `GET /api/models` - Ollama models
- `GET /health` - Health check
- `WS /ws/{client_id}` - Real-time chat

### Contributing
1. Fork repository
2. Create feature branch
3. Submit pull request

## License
MIT License