# Agent Labs Web App

A modern web application that provides a chat interface with Ollama language models, featuring an intelligent multi-agent framework with tool selection and execution capabilities.

## Features

- **Real-time Chat Interface**: WebSocket-based chat with streaming responses from Ollama models
- **Tool Selection Sidebar**: Choose from various categories of tools to enhance your AI assistant
- **Multi-Agent Framework**: Intelligent tool orchestration with specialized sub-agents and sequential dependency
- **Collapsible Tool Results**: Clean display with AI-generated summaries and expandable raw output
- **Multiple Models**: Support for various Ollama models (Gemma 3, Llama 3.1, Mistral, etc.)
- **Modern UI**: Clean, responsive interface built with React, Next.js, and Tailwind CSS

## Architecture

- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Backend**: FastAPI with WebSocket support and multi-agent system integration
- **LLM Integration**: Direct integration with Ollama API through custom OllamaLLM class
- **Agent Framework**: Multi-agent orchestrator with specialized sub-agents for each tool category
- **Real-time Communication**: WebSocket connections for streaming responses and tool execution

## Available Tools

### Filesystem
- **file_search**: Search for files in the filesystem

### Web
- **web_search**: Search the web for information

### Development
- **code_analysis**: Analyze code files for patterns and issues

### Data
- **data_processing**: Process and transform data

### System
- **system_info**: Get system information and metrics

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ollama installed locally (or use the Docker service)

### Using Docker Compose (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd agent_labs_ollama
```

2. Start all services:
```bash
docker-compose up --build
```

3. Open your browser and navigate to:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Ollama API: http://localhost:11434

### Manual Setup

#### Backend Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Navigate to the backend directory:
```bash
cd backend
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
python main.py
```

#### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

#### Ollama Setup

1. Install Ollama following the official instructions
2. Pull recommended models:
```bash
ollama pull gemma2:latest
ollama pull llama3.1:latest
ollama pull mistral:latest
```

3. Start Ollama server:
```bash
ollama serve
```

#### Google Search API Setup (Optional)

For enhanced web search functionality, configure Google Custom Search API:

1. **Get Google API Key:**
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing project
   - Enable Custom Search API
   - Create credentials (API Key)

2. **Create Custom Search Engine:**
   - Visit [Custom Search Engine](https://cse.google.com/cse/)
   - Create a new search engine
   - Configure to search the entire web
   - Get your Search Engine ID

3. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your credentials:
   GOOGLE_SEARCH_API_KEY=your_api_key_here
   GOOGLE_SEARCH_ENGINE_ID=your_engine_id_here
   ```

**Note:** Without Google API credentials, the web search tool will return an error message. The system will still function with other tools.

## Usage

1. **Select Tools**: Use the sidebar to select tools you want the AI to have access to
2. **Start Chatting**: Type your message in the chat interface
3. **Intelligent Tool Execution**: The multi-agent orchestrator will automatically analyze your request and execute appropriate tools with sequential dependency
4. **View Results**: Tool execution results are displayed with AI-generated summaries and collapsible raw output

## Example Interactions

**With system_info and web_search tools selected:**
```
User: "Find my system info and check online if there are newer available"
Orchestrator: Selecting SystemInfoAgent and WebSearchAgent for sequential execution.

SystemInfoAgent: Retrieving system overview...
Result: macOS Darwin 25.0.0, ARM64, 16GB RAM, 10 cores

WebSearchAgent: Searching for "macOS Darwin 25.0.0 updates" based on system info...
Result: Found information about macOS Tahoe updates and compatibility

Final Answer: Your system is running macOS Darwin 25.0.0 (Tahoe) with ARM64 architecture and 16GB RAM. Based on the search results, this appears to be a recent pre-release version with ongoing development updates.
```

**With code_analysis tool selected:**
```
User: "Analyze the security of my Python authentication code"
Orchestrator: Selecting CodeAnalysisAgent for security analysis.

CodeAnalysisAgent: Analyzing authentication code for security vulnerabilities...
Result: {"file_path": "auth.py", "analysis_type": "security", "findings": [...]}

Final Answer: Found 2 security issues: password hashing uses deprecated MD5 and no rate limiting on login attempts. Recommend using bcrypt and implementing exponential backoff.
```

## Development

### Backend Development

The backend is built with FastAPI and a custom multi-agent system and includes:
- WebSocket endpoint for real-time communication
- Multi-agent orchestrator with specialized sub-agents
- Custom OllamaLLM integration
- Sequential dependency coordination between agents
- Tool result summarization with AI
- CORS middleware for frontend communication

Key files:
- `main.py`: Main application with WebSocket and API endpoints
- `multi_agent_system.py`: Multi-agent orchestrator and specialized sub-agents
- `requirements.txt`: Python dependencies for multi-agent system

### Frontend Development

The frontend is built with Next.js 14 and includes:
- Real-time chat interface with WebSocket
- Tool selection sidebar with categories
- Collapsible tool results with AI summaries
- TypeScript for type safety
- Tailwind CSS for styling

Key files:
- `src/app/page.tsx`: Main application component
- `src/components/ChatInterface.tsx`: Chat UI component with collapsible results
- `src/components/ToolSidebar.tsx`: Tool selection component
- `src/hooks/useWebSocket.ts`: WebSocket communication hook with agent support

### Adding New Tools

1. **Create Tool Script**: Add a new Python script in the `tools/` directory
2. **Backend**: Add specialized agent class to `multi_agent_system.py`
3. **Frontend**: Tool will automatically appear in the sidebar

Example tool implementation:
```python
class NewToolAgent(BaseAgent):
    """Specialized agent for new tool operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute new tool with intelligent parameter extraction"""
        try:
            # Extract parameters from query using LLM
            params = self._extract_parameters(query)

            # Execute tool script
            result = self._execute_tool_script("new_tool", params)

            return {
                "agent": "NewToolAgent",
                "tool": "new_tool",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "agent": "NewToolAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
```

## API Endpoints

### REST API
- `GET /api/tools`: Get available tools
- `GET /api/models`: Get available Ollama models
- `POST /api/tool/execute`: Execute a specific tool
- `GET /health`: Health check

### WebSocket
- `WS /ws/{client_id}`: Real-time chat communication with agent orchestration

#### WebSocket Message Types
- `message_received`: Acknowledges user message
- `assistant_response_start`: Begin streaming response
- `assistant_response_chunk`: Response content chunk
- `assistant_response_complete`: Response finished
- `tool_execution_start`: Agent begins tool execution
- `tool_result`: Tool execution result
- `tool_summary`: AI-generated summary of tool result
- `agent_response`: Final agent conclusion
- `error`: Error message

## Configuration

### Environment Variables

**Backend:**
- `OLLAMA_BASE_URL`: Ollama API URL (default: http://localhost:11434)
- `GOOGLE_SEARCH_API_KEY`: Google Custom Search API key (optional)
- `GOOGLE_SEARCH_ENGINE_ID`: Google Custom Search Engine ID (optional)

**Frontend:**
- `NEXT_PUBLIC_API_URL`: Backend API URL
- `NEXT_PUBLIC_WS_URL`: WebSocket URL

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details