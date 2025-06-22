"""
Basic MCP Server implementation
"""
# Standard library imports
import json
from dataclasses import dataclass
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

# Third-party imports
from duckduckgo_search import DDGS


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str


class MCPServer:
    def __init__(self):
        self.tools: List[Tool] = []
        self.resources: List[Resource] = []
        self.setup_default_tools()
    
    def setup_default_tools(self):
        """Setup default tools for the MCP server"""
        self.tools.append(Tool(
            name="echo",
            description="Echo back the input text",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }
        ))
        
        self.tools.append(Tool(
            name="get_time",
            description="Get current system time",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        ))

        self.tools.append(Tool(
            name="search_web",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ))
    
    def handle_initialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request"""
        return {
            "server": {
                "name": "example-mcp-server",
                "version": "1.0.0",
                "protocolVersion": "2024-11-05"
            },
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.parameters
                }
                for tool in self.tools
            ]
        }
    
    def handle_list_tools(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools list request"""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.parameters
                }
                for tool in self.tools
            ]
        }
    
    def handle_call_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request"""
        tool_name = data.get("name")
        arguments = data.get("arguments", {})
        
        try:
            if tool_name == "echo":
                result = {"content": [{"type": "text", "text": arguments.get("message", "")}]}
            elif tool_name == "get_time":
                current_time = datetime.now()
                time_info = {
                    "timestamp": current_time.isoformat(),
                    "formatted": current_time.strftime("%B %d, %Y at %I:%M %p"),
                    "timezone": "local",
                    "components": {
                        "year": current_time.year,
                        "month": current_time.month,
                        "day": current_time.day,
                        "hour": current_time.hour,
                        "minute": current_time.minute,
                        "second": current_time.second
                    }
                }
                result = {"content": [{"type": "text", "text": json.dumps(time_info)}]}
            elif tool_name == "search_web":
                query = arguments.get("query", "")
                num_results = arguments.get("num_results", 3)
                result = self.search_web(query, num_results)
            else:
                return {"error": f"Tool '{tool_name}' not found"}
            
            return {"result": result}
            
        except Exception as e:
            print(f"Error in handle_call_tool: {str(e)}")
            return {"error": f"Internal error: {str(e)}"}
    
    def search_web(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Search the web using DuckDuckGo Search API"""
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": r.get("title"),
                        "content": r.get("body"),
                        "source": r.get("href")
                    })
            if not results:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "query": query,
                            "results": [{
                                "title": "No Direct Results",
                                "content": "I couldn't find specific information about this query. You might want to try rephrasing your question or being more specific.",
                                "source": "DuckDuckGo Search"
                            }]
                        })
                    }]
                }
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "query": query,
                        "results": results
                    })
                }]
            }
        except Exception as e:
            print(f"Error in search_web: {str(e)}")
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": f"Search failed: {str(e)}"
                    })
                }]
            }


class MCPRequestHandler(BaseHTTPRequestHandler):
    server_instance = None
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Get request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Get endpoint
            endpoint = self.path.strip('/')
            
            # Handle request based on endpoint
            if endpoint == "initialize":
                response = self.server_instance.handle_initialize(data)
            elif endpoint == "list_tools":
                response = self.server_instance.handle_list_tools(data)
            elif endpoint == "call_tool":
                response = self.server_instance.handle_call_tool(data)
            else:
                self.send_error(404, f"Endpoint '{endpoint}' not found")
                return
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON request")
        except Exception as e:
            self.send_error(500, f"Internal error: {str(e)}")
    
    def log_message(self, format, *args):
        """Override to use print instead of stderr"""
        print(format % args)


def run_server(port=8000):
    """Run the MCP server"""
    server = HTTPServer(('localhost', port), MCPRequestHandler)
    MCPRequestHandler.server_instance = MCPServer()
    print(f"MCP Server running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()