# MCP Agent Ollama

A sophisticated Python implementation that bridges Ollama's AI capabilities with the Model Control Protocol (MCP), enabling natural language interaction with structured tools and web services.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Interface │────▶│  MCP Client     │────▶│  MCP Server     │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                      │
                                 │                      │
                                 ▼                      ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  Ollama Client  │     │  Tool Handlers  │
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

## Core Components

### 1. **MCP Client** (`mcp_client.py`)
   - **Server Lifecycle Management**: Automated startup, monitoring, and cleanup of MCP server processes
   - **HTTP Request Handling**: Robust communication with MCP server using curl-based requests
   - **Configuration System**: JSON-based configuration with intelligent fallback defaults
   - **Comprehensive Testing**: Built-in test suite (`MCPTester`) for endpoint validation and reliability

### 2. **MCP Server** (`mcp_server.py`)
   - **Tool Implementation**: Three core tools (echo, get_time, search_web) with structured interfaces
   - **RESTful API**: HTTP server providing `/initialize`, `/list_tools`, and `/call_tool` endpoints
   - **Web Search Integration**: Real-time web search using DuckDuckGo Search API
   - **Error Handling**: Robust error recovery and fallback mechanisms

### 3. **Ollama Integration** (`mcp_ollama.py`)
   - **AI-Tool Bridge**: Intelligent parsing of Ollama responses to detect tool usage requirements
   - **System Prompt Engineering**: Sophisticated prompts that guide Ollama's tool selection decisions
   - **Response Processing**: Smart JSON parsing with error recovery for malformed responses
   - **Natural Language Generation**: Converts structured tool results into conversational responses

### 4. **Web Interface** (`app.py`)
   - **Modern Chat UI**: Streamlit-based interface with custom styling and avatars
   - **Real-time Interaction**: Asynchronous chat processing with live status updates
   - **Session Management**: Persistent chat history and server connection monitoring
   - **Professional Design**: Clean, responsive interface optimized for conversational AI

### 5. **Testing Suite** (`test_ollama.py`)
   - **Ollama Connectivity**: Validates Ollama server availability and model access
   - **Model Verification**: Confirms installed models and generation capabilities
   - **Integration Testing**: End-to-end testing of the complete system workflow

## Key Features

- 🤖 **AI-Powered Tool Integration**: Seamless bridge between Ollama's language models and structured tool execution
- 🔍 **Web Search Capabilities**: Real-time web search through DuckDuckGo integration
- ⏰ **System Information Access**: Get current time and system details through natural language
- 🧪 **Comprehensive Testing**: Extensive test suite ensuring reliability and performance
- 🔄 **Asynchronous Architecture**: Non-blocking operations for responsive user experience
- 📝 **Natural Language Processing**: Intelligent conversation flow with context awareness
- 🌐 **Modern Web Interface**: Professional Streamlit-based chat interface with real-time updates
- 💬 **Dual Interface Support**: Both command-line and web-based interaction modes
- ⚙️ **Flexible Configuration**: JSON-based settings for easy customization and deployment
- 🛡️ **Robust Error Handling**: Comprehensive error recovery and graceful fallback mechanisms

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp_agent_ollama
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

4. Start Ollama and pull a model:
```bash
# Start Ollama service
ollama serve

# In a new terminal, pull the Mistral model
ollama pull mistral
```

## Configuration

The project includes a `config.json` file with default settings:
```json
{
  "mcpServers": {
    "default": {
      "command": ["python", "mcp_server.py"],
      "url": "http://localhost:8000"
    }
  },
  "ollama": {
    "baseUrl": "http://localhost:11434",
    "model": "mistral:latest"
  }
}
```

**Configuration Options:**
- **mcpServers.default.command**: Command to start the MCP server
- **mcpServers.default.url**: URL where the MCP server will run  
- **ollama.baseUrl**: Ollama API endpoint (default: localhost:11434)
- **ollama.model**: Ollama model to use (supports: mistral, llama3, codellama, etc.)

## Usage

### Web Interface (Recommended)
Launch the modern web-based chat interface:
```bash
streamlit run app.py
```

**Web Interface Features:**
- 💬 **Interactive Chat**: Real-time conversation with the AI agent
- 📊 **Server Status**: Live monitoring of MCP client connection status  
- 🕒 **Chat History**: Persistent conversation history within sessions
- 🗑️ **Clear Chat**: Reset conversation with a single click
- 🎨 **Modern Design**: Professional, responsive interface with custom styling
- 👤 **Avatar Support**: Visual distinction between user and assistant messages

### Command Line Interface
For terminal-based interaction:
```bash
python mcp_ollama.py
```

**CLI Capabilities:**
- 🔍 **Natural Language Queries**: Ask questions and get AI-powered responses
- 🌐 **Automatic Web Search**: The AI will search the web when needed for current information
- ⏰ **System Information**: Get current time and system details
- 🔄 **Tool Integration**: The AI automatically selects and uses appropriate tools
- 🚪 **Clean Exit**: Type 'exit' or 'quit' to terminate the session

## Testing

### MCP System Testing
Run the comprehensive MCP test suite:
```bash
python mcp_client.py --test
```

**Test Coverage:**
- ✅ Server initialization and startup
- ✅ Tool discovery and listing  
- ✅ Tool execution (echo, get_time, search_web)
- ✅ Error handling and recovery
- ✅ HTTP endpoint validation

### Ollama Integration Testing
Test Ollama connectivity and functionality:
```bash
python test_ollama.py
```

**Validation Checks:**
- 🔌 Ollama server connectivity
- 📋 Available model verification
- 🤖 Text generation capabilities

## Available Tools

The MCP server provides three core tools accessible through natural language:

### 🔍 **Web Search** (`search_web`)
- **Purpose**: Search the web for current information
- **Usage**: "Search for the latest news about AI" or "Find information about Python async programming"
- **Parameters**: Query string and optional result count (default: 5)
- **Response**: Formatted search results with titles, descriptions, and URLs

### ⏰ **Time Information** (`get_time`) 
- **Purpose**: Get current system time and date information
- **Usage**: "What time is it?" or "What's the current date?"
- **Response**: Detailed time information including timezone, day of week, and formatted timestamps

### 📝 **Echo Tool** (`echo`)
- **Purpose**: Simple text echoing for testing and validation
- **Usage**: "Echo this message" or "Repeat what I say"
- **Response**: Returns the input text for testing purposes

## Troubleshooting

### OpenSSL Warning
If you encounter this warning:
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+
```

**Solutions:**
1. **Downgrade urllib3** (recommended):
   ```bash
   pip install "urllib3<2.0.0"
   ```

2. **Install newer OpenSSL**:
   ```bash
   # macOS
   brew install openssl@1.1
   
   # Ubuntu/Debian  
   sudo apt-get install libssl-dev
   ```

### Common Issues
- **Ollama Connection**: Ensure Ollama is running on `localhost:11434`
- **Model Missing**: Install required model with `ollama pull mistral`
- **Port Conflicts**: Check if port 8000 (MCP server) is available
- **Python Version**: Requires Python 3.7+ for asyncio support

## Project Structure

```
mcp_agent_ollama/
├── mcp_client.py       # MCP client with server lifecycle management
├── mcp_server.py       # HTTP server providing tool implementations  
├── mcp_ollama.py       # Ollama integration and conversation flow
├── app.py              # Streamlit web interface
├── test_ollama.py      # Ollama connectivity testing
├── config.json         # Configuration settings
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Guidelines:**
- Follow Python PEP 8 style conventions
- Add tests for new functionality
- Update documentation for API changes
- Ensure compatibility with Python 3.7+
