"""
MCP Client that connects with Ollama
"""

# Standard library imports
import asyncio
import json
from typing import Dict, Any, List, Optional

# Third-party imports
import requests

# Local imports
from mcp_client import MCPClient


class OllamaClient:
    def __init__(self, base_url: str = None, model: str = None):
        # Create MCPClient instance to load config
        mcp_client = MCPClient()
        config = mcp_client.config
        self.base_url = base_url or config.get("ollama", {}).get("baseUrl", "http://localhost:11434")
        self.model = model or config.get("ollama", {}).get("model", "mistral:latest")
        self.stream = False  # Default to non-streaming mode
    
    def generate(self, prompt: str, stream: bool = None) -> str:
        """Generate text using Ollama"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream if stream is not None else self.stream
            }
            
            print(f"Calling Ollama at: {url} with model: {self.model}")
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    full_response += data['response']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError as e:
                                print(f"Warning: Skipping invalid JSON in stream: {e}")
                                continue
                    return full_response
                else:
                    try:
                        return response.json().get('response', '')
                    except json.JSONDecodeError as e:
                        print(f"Error parsing non-streaming response: {e}")
                        print(f"Raw response: {response.text}")
                        return "Error: Could not parse response from Ollama"
            else:
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure 'ollama serve' is running.")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. The model might be loading.")
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat with Ollama using conversation format"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                try:
                    return response.json().get('message', {}).get('content', '')
                except json.JSONDecodeError as e:
                    print(f"Error parsing chat response: {e}")
                    print(f"Raw response: {response.text}")
                    return "Error: Could not parse response from Ollama"
            else:
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure 'ollama serve' is running.")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. The model might be loading.")


class MCPOllamaIntegration:
    def __init__(self, server_name: str = "default", ollama_base_url: str = None, ollama_model: str = None):
        # Create MCPClient instance to load config
        self.mcp_client = MCPClient()
        config = self.mcp_client.config
        self.ollama_client = OllamaClient(
            ollama_base_url or config.get("ollama", {}).get("baseUrl", "http://localhost:11434"),
            ollama_model or config.get("ollama", {}).get("model", "mistral:latest")
        )
        self.available_tools = []
        self.config = config
        self.conversation_history = []
    
    def get_system_prompt(self):
        """Get the system prompt for Ollama"""
        return """You are a helpful AI assistant that can use tools to help users. You have access to the following tools:

1. echo: Echo back the input text
   - Parameters:
     - message (string): The message to echo back

2. get_time: Get current system time
   - Parameters: None

3. search_web: Search the web for information
   - Parameters:
     - query (string): The search query
     - num_results (integer, optional): Number of results to return (default: 3)

To use a tool, respond with a JSON object in the following format:
{
    "use_tool": {
        "name": "tool_name",
        "arguments": {
            "param1": "value1",
            "param2": "value2"
        }
    }
}

For example, to search the web:
{
    "use_tool": {
        "name": "search_web",
        "arguments": {
            "query": "Python programming",
            "num_results": 2
        }
    }
}

Always respond with valid JSON. If you need to provide additional context or explanation, include it in a "message" field in your response."""

    async def handle_tool_response(self, tool_name: str, response: Dict[str, Any]) -> str:
        """Handle tool response and format it for the user"""
        if tool_name == "echo":
            return response.get("response", "")
        elif tool_name == "get_time":
            try:
                time_info = json.loads(response.get("result", {}).get("content", [{}])[0].get("text", "{}"))
                if "formatted" in time_info:
                    return f"It is currently {time_info['formatted']}."
                else:
                    # Fallback to timestamp if formatted time is not available
                    return f"It is currently {time_info.get('timestamp', 'unknown time')}."
            except (json.JSONDecodeError, KeyError, IndexError):
                return "I'm sorry, I couldn't get the current time."
        elif tool_name == "search_web":
            # Extract results from the response
            result_content = response.get("result", {}).get("content", [{}])[0].get("text", "{}")
            try:
                search_data = json.loads(result_content)
                results = search_data.get("results", [])
                
                if not results:
                    return "I couldn't find any specific information about that. Could you please rephrase your question or provide more details?"
                
                # Format the results
                formatted_results = []
                for result in results:
                    title = result.get("title", "N/A")
                    content = result.get("content", "N/A")
                    source = result.get("source", "N/A")
                    
                    # Skip error messages and empty results
                    if title in ["Error", "No Direct Results"]:
                        continue
                        
                    formatted_results.append(
                        f"📌 {title}\n"
                        f"{content}\n"
                        f"Source: {source}\n"
                    )
                
                if not formatted_results:
                    return "I couldn't find any specific information about that. Could you please rephrase your question or provide more details?"
                
                return "\n---\n".join(formatted_results)
                
            except json.JSONDecodeError:
                return "I encountered an error while processing the search results. Please try again."
        else:
            return json.dumps(response, indent=2)

    async def process_ollama_response(self, response: str) -> str:
        """Process response from Ollama"""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            # Check if it's a tool call
            if "use_tool" in data:
                tool_call = data["use_tool"]
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                
                # Call the tool
                tool_response = await self.mcp_client.call_tool(tool_name, arguments)
                
                # Format the tool response
                formatted_response = await self.handle_tool_response(tool_name, tool_response)
                
                # If there's a message, include it
                if "message" in data:
                    return f"{data['message']}\n\nTool Response:\n{formatted_response}"
                return formatted_response
            
            # If it's a regular message
            return data.get("message", response)
            
        except json.JSONDecodeError:
            # If not JSON, return as is
            return response

    async def initialize(self):
        """Initialize both MCP and Ollama connections"""
        await self.mcp_client.start_server()
        response = await self.mcp_client.list_tools()
        self.available_tools = response.get("tools", [])
        print(f"Available MCP tools: {[tool['name'] for tool in self.available_tools]}")
    
    def create_system_prompt(self) -> str:
        """Create system prompt that includes available tools"""
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.available_tools
        ])
        
        return f"""You are an AI assistant with access to the following tools:

{tools_description}

CRITICAL RESPONSE RULES:

1. For queries requiring external information (ALWAYS use search_web):
   - ANY question about people, places, current events, or facts
   - ANY question that might need up-to-date information
   - ANY question you're not 100% certain about
   - Respond with ONLY the search_web tool JSON
   - NO natural language response
   - NO explanations about using tools
   - Example: {{"use_tool": {{"name": "search_web", "arguments": {{"query": "Redmond"}}}}}}

2. For direct answers (ONLY for general knowledge you're certain about):
   - Respond with natural language text ONLY
   - NEVER include any JSON or tool usage examples
   - NEVER mention tools or tool usage
   - Example: "I understand you're asking about Python. Python is a high-level programming language..."

3. For other tool usage (get_time, echo):
   - Respond with ONLY a JSON object in this exact format:
   {{"use_tool": {{"name": "tool_name", "arguments": {{"param": "value"}}}}}}
   - NO natural language text
   - NO explanations
   - NO tool usage examples

Remember:
- For ANY external information, use search_web ONLY
- NEVER provide natural language before using search_web
- NEVER explain that you're using a tool
- NEVER mix JSON and natural language
- NEVER include tool usage examples"""
    
    async def process_user_input(self, user_query: str) -> str:
        """Process user query, potentially using MCP tools"""
        system_prompt = self.create_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser: {user_query}\nAssistant:"
        
        # Get response from Ollama
        ollama_response = self.ollama_client.generate(full_prompt)
        print(f"\n[Ollama] Response: {ollama_response}")
        
        # Check if Ollama wants to use a tool
        try:
            response_text = ollama_response.strip()
            
            # Try to parse as JSON tool request
            if response_text.startswith('{'):
                # Clean up the response text
                response_text = response_text.replace(' {', '{').replace('{ ', '{')
                # Remove any trailing quotes or invalid characters
                response_text = response_text.rstrip('"\'')
                
                if '"use_tool"' in response_text:
                    try:
                        tool_request = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        print(f"[Error] JSON decode error: {e}")
                        # Try to extract tool information even if JSON is malformed
                        if '"use_tool"' in response_text and '"name"' in response_text and '"arguments"' in response_text:
                            # Extract tool name and arguments using string manipulation
                            try:
                                tool_name_start = response_text.find('"name":') + 7
                                tool_name_end = response_text.find('"', tool_name_start)
                                tool_name = response_text[tool_name_start:tool_name_end].strip('" ')
                                
                                if tool_name in [tool['name'] for tool in self.available_tools]:
                                    print(f"[Tool] Using recovered tool: {tool_name}")
                                    # Call the MCP tool with empty arguments
                                    tool_response = await self.mcp_client.call_tool(tool_name, {})
                                    tool_result = tool_response.get("result", {}).get("content", [{}])[0].get("text", "")
                                    print(f"[Tool] Response: {tool_result}")
                                    
                                    final_prompt = f"User asked: {user_query}\n\nI used the {tool_name} tool and got this result: {tool_result}\n\nNow provide a natural, helpful response to the user based on this information. Use natural language only, no JSON:"
                                    final_response = self.ollama_client.generate(final_prompt)
                                    return final_response
                            except Exception as e:
                                print(f"[Error] Failed to recover tool information: {e}")
                        return ollama_response
                    
                    tool_call = tool_request.get("use_tool", {})
                    tool_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})
                    
                    if tool_name and tool_name in [tool['name'] for tool in self.available_tools]:
                        print(f"[Tool] Using tool: {tool_name}")
                        print(f"[Tool] Arguments: {json.dumps(arguments, indent=2)}")
                        
                        # Call the MCP tool
                        tool_response = await self.mcp_client.call_tool(tool_name, arguments)
                        tool_result = tool_response.get("result", {}).get("content", [{}])[0].get("text", "")
                        print(f"[Tool] Response: {tool_result}")
                        
                        # For search results, format them nicely
                        if tool_name == "search_web":
                            try:
                                search_data = json.loads(tool_result)
                                results = search_data.get("results", [])
                                
                                if not results:
                                    return "I couldn't find any specific information about that. Could you please rephrase your question or provide more details?"
                                
                                # Format the results
                                formatted_results = []
                                for result in results:
                                    title = result.get("title", "N/A")
                                    content = result.get("content", "N/A")
                                    source = result.get("source", "N/A")
                                    
                                    # Skip error messages and empty results
                                    if title in ["Error", "No Direct Results"]:
                                        continue
                                        
                                    formatted_results.append(
                                        f"📌 {title}\n"
                                        f"{content}\n"
                                        f"Source: {source}\n"
                                    )
                                
                                if not formatted_results:
                                    return "I couldn't find any specific information about that. Could you please rephrase your question or provide more details?"
                                
                                # Generate a natural response based on the search results
                                final_prompt = f"""User asked: {user_query}

I found this information:
{chr(10).join(formatted_results)}

Based on this information, provide a natural, helpful response to the user. 
Use natural language only, no JSON. 
If the information seems outdated or incomplete, mention that.
If there are multiple relevant pieces of information, synthesize them into a coherent response."""

                                final_response = self.ollama_client.generate(final_prompt)
                                return final_response
                                
                            except json.JSONDecodeError:
                                return "I encountered an error while processing the search results. Please try again."
                        
                        # For other tools, generate a natural response
                        final_prompt = f"User asked: {user_query}\n\nI used the {tool_name} tool and got this result: {tool_result}\n\nNow provide a natural, helpful response to the user based on this information. Use natural language only, no JSON:"
                        final_response = self.ollama_client.generate(final_prompt)
                        return final_response
                    else:
                        print(f"[Error] Invalid tool name: {tool_name}")
                        return ollama_response
            else:
                # Not a tool request, return the original response
                return ollama_response
                
        except Exception as e:
            print(f"[Error] Failed to process tool request: {e}")
            return ollama_response
    
    def close(self):
        """Clean up connections"""
        self.mcp_client.close()


async def main():
    """Main function to run the MCP client"""
    config = MCPClient().config
    
    # Use default server from config
    server_name = "default"
    if "mcpServers" in config and "default" not in config["mcpServers"]:
        # If no default server, use the first available server
        available_servers = list(config["mcpServers"].keys())
        if available_servers:
            server_name = available_servers[0]
        else:
            raise ValueError("No MCP servers configured in config.json")
    
    print(f"\nUsing MCP server: {server_name}")
    
    # Create integration
    integration = MCPOllamaIntegration(
        server_name=server_name,
        ollama_model=config["ollama"]["model"]
    )
    
    # Initialize
    await integration.initialize()
    
    # Main interaction loop
    print("\nMCP Client is ready! Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                break
            
            response = await integration.process_user_input(user_input)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main()) 