# MCP (Model Control Protocol) Client

A Python implementation of the Model Control Protocol client that integrates with Ollama for AI-powered tool usage.

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

## Components

1. **MCP Client** (`mcp_client.py`)
   - Manages server lifecycle
   - Handles tool requests and responses
   - Provides testing framework

2. **MCP Server** (`mcp_server.py`)
   - Implements tool handlers
   - Manages HTTP endpoints
   - Provides web search functionality

3. **Ollama Integration** (`mcp_ollama.py`)
   - Connects to Ollama API
   - Processes user queries
   - Manages conversation flow

4. **Web Interface** (`app.py`)
   - Streamlit-based chat interface
   - Real-time message updates
   - Modern, responsive design
   - Server status monitoring

## Features

- 🤖 AI-powered tool usage with Ollama
- 🔍 Web search capabilities
- ⏰ Time and date information
- 🧪 Comprehensive test suite
- 🔄 Asynchronous operations
- 📝 Natural language responses
- 🌐 Web-based chat interface
- 💬 Real-time conversation flow

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp-client
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

4. Start Ollama:
```bash
ollama serve
```

5. Pull the Mistral model:
```bash
ollama pull mistral
```

## Configuration

Create a `config.json` file:
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

## Usage

### Web Interface
Start the Streamlit web app:
```bash
streamlit run app.py
```

The web interface provides:
- Real-time chat with the AI
- Server status monitoring
- Chat history
- Clear chat functionality
- Modern, responsive design

### Command Line Interface
Start the CLI client:
```bash
python mcp_ollama.py
```

Available commands:
- Ask questions (uses web search)
- Get current time
- Echo messages
- Type 'exit' to quit

## Testing

Run the test suite:
```bash
python mcp_client.py --test
```

## OpenSSL Warning

If you see this warning:
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'
```

You have two options:

1. Downgrade urllib3:
```bash
pip install urllib3==1.26.x
```

2. Install OpenSSL 1.1.1 or later:
```bash
# macOS
brew install openssl@1.1

# Ubuntu/Debian
sudo apt-get install libssl1.1
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
