# Agent Labs

A modern AI-powered chat interface with intelligent multi-agent orchestration and real-time tool execution. Built with Next.js, FastAPI, and Ollama integration.

## Features

- **Real-time Chat Interface**: WebSocket-based streaming communication with immediate response feedback
- **Intelligent Multi-Agent System**: Orchestrator pattern with specialized sub-agents for different tool categories
- **Tool Selection & Execution**: Dynamic tool selection with automatic parameter extraction and execution
- **File Upload Support**: Analyze images, process data files, and generate presentations from uploaded content
- **Multiple Model Support**: Compatible with various Ollama models (Gemma, Llama, Mistral, etc.)
- **Modern Responsive UI**: Clean interface built with React, Next.js, and Tailwind CSS
- **Real-time Tool Results**: Live updates with collapsible results and AI-generated summaries

## Architecture

### System Design

```
┌─────────────────┐    WebSocket    ┌──────────────────┐    HTTP/API    ┌─────────────┐
│  Next.js        │◄──────────────►│  FastAPI         │◄──────────────►│  Ollama     │
│  Frontend       │                │  Backend         │                │  LLM        │
│                 │                │                  │                │             │
│ - Chat UI       │                │ - WebSocket      │                │ - Models    │
│ - Tool Sidebar  │                │ - Multi-Agent    │                │ - Chat API  │
│ - File Upload   │                │ - Orchestrator   │                │             │
└─────────────────┘                └──────────────────┘                └─────────────┘
                                            │
                                            │
                                    ┌───────▼────────┐
                                    │  Sub-Agents    │
                                    │                │
                                    │ - FileSearch   │
                                    │ - WebSearch    │
                                    │ - SystemInfo   │
                                    │ - Presentation │
                                    │ - CodeAnalysis │
                                    │ - ImageAnalysis│
                                    │ - DataProcess  │
                                    │ - StockAnalysis│
                                    └────────────────┘
```

### Communication Flow

1. **User Input** → Frontend captures message and selected tools
2. **WebSocket** → Real-time bidirectional communication
3. **Orchestrator** → Analyzes query and selects appropriate sub-agents
4. **Initial Response** → Immediate acknowledgment sent to user
5. **Tool Execution** → Sub-agents execute tools with extracted parameters
6. **Real-time Updates** → Tool results streamed back to frontend
7. **Final Synthesis** → Orchestrator combines results into comprehensive answer

### Multi-Agent Pattern

- **OrchestratorAgent**: Main coordinator that selects and manages sub-agents
- **Specialized Sub-Agents**: Each handles specific tool categories with domain expertise
- **Context Sharing**: Subsequent agents receive results from previous executions
- **Sequential Dependencies**: Agents can build upon each other's results

## Project Structure

```
agent_labs_ollama/
├── backend/
│   ├── main.py                     # FastAPI app with WebSocket endpoints
│   ├── multi_agent_system.py       # Main multi-agent system interface
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Base agent class with LLM integration
│   │   ├── orchestrator_agent.py   # Main orchestrator with callback support
│   │   ├── file_search_agent.py    # File system search operations
│   │   ├── web_search_agent.py     # Web search with Google API
│   │   ├── system_info_agent.py    # System metrics and information
│   │   ├── code_analysis_agent.py  # Code quality and security analysis
│   │   ├── data_processing_agent.py # Data analysis and transformation
│   │   ├── presentation_agent.py   # PowerPoint generation
│   │   ├── image_analysis_agent.py # Image content analysis
│   │   └── stock_analysis_agent.py  # Stock market analysis
│   ├── tools/
│   │   ├── file_search.py          # File system search implementation
│   │   ├── web_search.py           # Google Custom Search integration
│   │   ├── system_info.py          # System metrics collection
│   │   ├── code_analysis.py        # Code analysis utilities
│   │   ├── data_processing.py      # Data manipulation tools
│   │   ├── presentation.py         # PowerPoint generation
│   │   ├── image_analysis.py       # Image processing and analysis
│   │   └── stock_analysis.py       # Stock market data analysis
│   └── requirements.txt            # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Main application page
│   │   │   ├── layout.tsx          # Root layout
│   │   │   └── globals.css         # Global styles
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx   # Main chat UI with file upload
│   │   │   ├── ToolSidebar.tsx     # Tool selection interface
│   │   │   ├── MessageBubble.tsx   # Individual message display
│   │   │   └── ToolResult.tsx      # Collapsible tool results
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts     # WebSocket communication
│   │   │   └── messageReducer.ts   # Message state management
│   │   └── types/
│   │       └── index.ts            # TypeScript type definitions
│   ├── package.json                # Node.js dependencies
│   └── tailwind.config.js          # Tailwind CSS configuration
├── docker-compose.yml              # Multi-service Docker setup
├── Dockerfile.backend              # Backend container configuration
├── Dockerfile.frontend             # Frontend container configuration
└── README.md                       # Project documentation
```

## Available Tools

The tools are organized into two main categories accessible via the sidebar:

### General Tools
- **file_search**: Search for files and directories in the filesystem
  - Supports pattern matching and recursive search
  - Returns file paths, sizes, and modification dates

- **web_search**: Search the internet for current information
  - Google Custom Search API integration
  - Real-time web results with relevance ranking

- **system_info**: Comprehensive system metrics
  - CPU, memory, disk usage, and network information
  - Operating system details and hardware specifications

- **presentation**: Generate PowerPoint presentations
  - Intelligent slide creation using Ollama LLM analysis
  - Template-based formatting and layout
  - Support for text files and data sources

### Analytics & Data Tools
- **code_analysis**: Analyze code files for quality and security
  - Security vulnerability detection
  - Code quality metrics and performance analysis
  - Support for multiple programming languages

- **image_analysis**: Analyze uploaded images
  - Object detection and scene recognition
  - Text extraction (OCR) from images
  - Image metadata and technical analysis

- **data_processing**: Process and analyze data files
  - CSV, JSON, and structured data analysis
  - Statistical operations and data transformation
  - Text analysis and pattern extraction

- **stock_analysis**: Analyze stock market data and performance
  - Real-time stock data from Yahoo Finance
  - Technical analysis with indicators (RSI, moving averages, Bollinger Bands)
  - Risk metrics and investment recommendations
  - AI-powered market insights

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd agent_labs_ollama
```

2. **Start all services**:
```bash
docker-compose up --build
```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Ollama API: http://localhost:11434

### Manual Setup

#### Prerequisites
- Python 3.9+
- Node.js 18+
- Ollama installed locally

#### Backend Setup

1. **Create virtual environment**:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start the server**:
```bash
python main.py
```

#### Frontend Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

2. **Start development server**:
```bash
npm run dev
```

#### Ollama Setup

1. **Install Ollama** following [official instructions](https://ollama.ai/)

2. **Pull recommended models**:
```bash
ollama pull gemma2:latest
ollama pull llama3.1:latest
ollama pull mistral:latest
```

3. **Start Ollama server**:
```bash
ollama serve
```

### Optional: Google Search API

For enhanced web search capabilities:

1. **Get Google API credentials**:
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Custom Search API
   - Create API key and Search Engine ID

2. **Configure environment**:
```bash
# Backend .env file
GOOGLE_SEARCH_API_KEY=your_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_engine_id_here
```

## Usage

### Basic Chat
1. Open the application at http://localhost:3000
2. Select tools from the sidebar based on your needs
3. Type your message and press Enter
4. Watch as the AI orchestrator selects and executes appropriate tools

### File Upload
1. Click the attachment icon in the chat input
2. Select an image, document, or data file
3. The system automatically chooses the appropriate analysis tool
4. Results include both analysis and file-specific insights

### Example Interactions

**System Analysis**:
```
User: "What are my system specs and are there any performance issues?"

Orchestrator: I'll check your system information and analyze performance.

SystemInfoAgent: Retrieved system overview - macOS Darwin 25.0.0, ARM64, 16GB RAM
Result: CPU usage at 15%, Memory 8.2GB used (51%), Disk 85% full

Final Answer: Your MacBook Air is running well with moderate resource usage.
However, disk space is getting low at 85% capacity. Consider cleaning up
large files or moving data to external storage.
```

**Code Security Review**:
```
User: "Analyze the security of my authentication module"

Orchestrator: I'll perform a security analysis of your authentication code.

CodeAnalysisAgent: Scanning for security vulnerabilities...
Result: Found 3 issues: weak password hashing, missing rate limiting,
SQL injection vulnerability in login query

Final Answer: Critical security issues found. Recommend: 1) Upgrade to
bcrypt for password hashing, 2) Implement rate limiting with exponential
backoff, 3) Use parameterized queries to prevent SQL injection.
```

**Stock Market Analysis**:
```
User: "Analyze Apple stock performance over the last year"

Orchestrator: I'll analyze AAPL stock performance using market data.

StockAnalysisAgent: Retrieving Yahoo Finance data for AAPL...
Result: Current price $189.50 (+12.3% YTD), RSI: 65 (neutral),
20-day MA: $185.20, Volatility: 23.4%, Recommendation: HOLD

AI Insights: Apple shows strong fundamentals with steady growth.
Recent momentum is positive but approaching overbought territory.
Consider taking profits if holding large positions.

Final Answer: Apple stock has performed well with 12.3% gains this year.
Technical indicators suggest neutral sentiment with some upward momentum.
The stock is fairly valued at current levels around $189.50.
```

## API Reference

### REST Endpoints

- `GET /api/tools` - List available tools
- `GET /api/models` - List available Ollama models
- `GET /health` - Health check

### WebSocket Protocol

Connect to `/ws/{client_id}` for real-time communication.

**Message Types**:
- `message_received` - Acknowledges user input
- `assistant_response_start` - Begin streaming response
- `assistant_response_chunk` - Response content chunk
- `assistant_response_complete` - Response finished
- `tool_result` - Tool execution result with raw data
- `tool_summary` - AI-generated summary of tool result
- `error` - Error message

**Message Format**:
```json
{
  "type": "assistant_response_chunk",
  "content": "I'll analyze your system...",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Configuration

### Environment Variables

**Backend** (`.env`):
```env
OLLAMA_BASE_URL=http://localhost:11434
GOOGLE_SEARCH_API_KEY=optional_api_key
GOOGLE_SEARCH_ENGINE_ID=optional_engine_id
LOG_LEVEL=INFO
```

**Frontend** (`.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Model Configuration

The system supports any Ollama-compatible model. Configure in the frontend model selector or set default in backend configuration.

## Development

### Adding New Tools

1. **Create tool implementation** in `backend/tools/new_tool.py`
2. **Create specialized agent** in `backend/agents/new_tool_agent.py`
3. **Register agent** in `backend/agents/__init__.py`
4. **Add to orchestrator** in `backend/agents/orchestrator_agent.py`

**Example tool structure**:
```python
class NewToolAgent(BaseAgent):
    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Extract parameters using LLM
            params = self._extract_parameters(query)

            # Execute tool
            result = self._execute_tool_script("new_tool", params)

            return {
                "agent": "NewToolAgent",
                "tool": "new_tool",
                "success": True,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
```

### Frontend Development

The frontend uses modern React patterns:
- **TypeScript** for type safety
- **Custom hooks** for WebSocket communication
- **useReducer** for complex state management
- **Tailwind CSS** for styling

### Backend Development

The backend implements:
- **AsyncIO** for concurrent operations
- **WebSocket** for real-time communication
- **Pydantic** for data validation
- **Multi-agent pattern** for tool orchestration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.