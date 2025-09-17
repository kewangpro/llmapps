from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import httpx
import asyncio
from datetime import datetime
import uuid
import base64
import tempfile
import os
from multi_agent_system import MultiAgentSystem

app = FastAPI(title="Agent Labs", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"Failed to send message to {client_id}: {e}")
                # Remove the disconnected client
                self.disconnect(client_id)

manager = ConnectionManager()

# Ollama client configuration
OLLAMA_BASE_URL = "http://localhost:11434"
ollama_client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=30.0)

# Tool registry
class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str

# Sample tools
AVAILABLE_TOOLS = {
    "file_search": Tool(
        name="file_search",
        description="Search for files in the filesystem",
        parameters={"path": "string", "pattern": "string"},
        category="filesystem"
    ),
    "web_search": Tool(
        name="web_search",
        description="Search the web for information",
        parameters={"query": "string", "limit": "number"},
        category="web"
    ),
    "code_analysis": Tool(
        name="code_analysis",
        description="Analyze code files for patterns and issues",
        parameters={"file_path": "string", "analysis_type": "string"},
        category="development"
    ),
    "data_processing": Tool(
        name="data_processing",
        description="Process and transform data",
        parameters={"input_data": "string", "operation": "string"},
        category="data"
    ),
    "system_info": Tool(
        name="system_info",
        description="Get system information and metrics",
        parameters={"metric": "string"},
        category="system"
    )
}

# Initialize Multi-Agent System
multi_agent_system = None

def get_multi_agent_system(model: str = "gemma3:latest") -> MultiAgentSystem:
    """Get or create multi-agent system instance"""
    global multi_agent_system
    if multi_agent_system is None:
        multi_agent_system = MultiAgentSystem(model=model)
    return multi_agent_system

# Ollama integration
async def stream_ollama_response(messages: List[Dict], model: str = "gemma3:latest"):
    """Stream response from Ollama"""
    try:
        async with ollama_client.stream(
            'POST',
            '/api/chat',
            json={
                "model": model,
                "messages": messages,
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if not chunk.get('done', False):
                            yield chunk.get('message', {}).get('content', '')
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"Error: {str(e)}"

def handle_attached_file(attached_file_data: Dict[str, Any], file_content: str) -> Optional[Dict[str, str]]:
    """Handle attached file by saving it to a temporary location"""
    if not attached_file_data or not file_content:
        return None

    try:
        file_name = attached_file_data.get("name", "unknown_file")
        file_type = attached_file_data.get("type", "")

        # Check if it's an image file sent as base64
        if file_type.startswith("image/") and file_content.startswith("data:"):
            # Extract base64 data from data URL
            header, base64_data = file_content.split(",", 1)

            # Decode base64 to binary
            file_binary = base64.b64decode(base64_data)

            # Create temporary file
            suffix = os.path.splitext(file_name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_binary)
                temp_path = temp_file.name

            return {
                "name": file_name,
                "path": temp_path,
                "type": file_type
            }
        else:
            # For text files, save the content directly
            suffix = os.path.splitext(file_name)[1] or ".txt"
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix, encoding='utf-8') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            return {
                "name": file_name,
                "path": temp_path,
                "type": file_type
            }
    except Exception as e:
        print(f"Error handling attached file: {e}")
        return None

def get_ollama_summary(tool_name: str, tool_result: Dict[str, Any], model: str = "gemma3:latest") -> str:
    """Get a summary of tool results from Ollama"""
    try:
        prompt = f"""Please provide a concise, user-friendly summary of this tool execution result. Focus on the key information that would be useful to the user.

Tool: {tool_name}
Result: {json.dumps(tool_result, indent=2)}

Provide a brief, clear summary in 1-2 sentences highlighting the most important findings or data."""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes tool execution results in a clear, concise manner."},
            {"role": "user", "content": prompt}
        ]

        import httpx
        with httpx.Client(base_url="http://localhost:11434", timeout=30.0) as client:
            response = client.post(
                '/api/chat',
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
            )
            result = response.json()
            return result.get('message', {}).get('content', 'Summary not available')

    except Exception as e:
        return f"Could not generate summary: {str(e)}"

# Chat message model
class ChatMessage(BaseModel):
    content: str
    role: str = "user"
    tools: Optional[List[str]] = None

# API endpoints
@app.get("/api/tools")
async def get_tools():
    """Get available tools"""
    mas = get_multi_agent_system()
    tools = mas.get_available_tools()

    # Convert to the format expected by frontend
    tools_dict = {}
    for tool in tools:
        tools_dict[tool["name"]] = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {},
            "category": "general"
        }

    return {"tools": tools_dict}

@app.get("/api/models")
async def get_models():
    """Get available Ollama models"""
    try:
        response = await ollama_client.get("/api/tags")
        return response.json()
    except Exception as e:
        return {"error": str(e), "models": []}

@app.post("/api/tool/execute")
async def execute_tool(tool_name: str, parameters: Dict[str, Any]):
    """Execute a specific tool"""
    result = await agent_framework.execute_tool(tool_name, parameters)
    return result

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_message = message_data.get("message", "")
            selected_tools = message_data.get("tools", [])
            model = message_data.get("model", "gemma3:latest")
            attached_file_data = message_data.get("attachedFile", None)

            # Debug: Log ALL received data
            print(f"DEBUG: Full message data keys: {list(message_data.keys())}")
            print(f"DEBUG: Message: '{user_message}'")
            print(f"DEBUG: Tools: {selected_tools}")
            print(f"DEBUG: Model: {model}")
            print(f"DEBUG: Received attachedFile data: {attached_file_data}")
            print(f"DEBUG: AttachedFile type: {type(attached_file_data)}")

            if attached_file_data:
                print(f"DEBUG: File name: {attached_file_data.get('name')}")
                print(f"DEBUG: File type: {attached_file_data.get('type')}")
                print(f"DEBUG: File size: {attached_file_data.get('size')}")
                content = attached_file_data.get("content", "")
                print(f"DEBUG: Content length: {len(content) if content else 0}")
                print(f"DEBUG: Content preview: {content[:50]}..." if content else "DEBUG: No content")

            # Process attached file if present
            attached_file = None
            if attached_file_data and attached_file_data.get("content"):
                print("DEBUG: Processing attached file...")
                attached_file = handle_attached_file(attached_file_data, attached_file_data.get("content"))
                print(f"DEBUG: Processed file result: {attached_file}")
            else:
                print("DEBUG: No attached file or content to process")

            # Send acknowledgment
            await manager.send_personal_message({
                "type": "message_received",
                "timestamp": datetime.now().isoformat()
            }, client_id)

            # Prepare system message with tool context
            system_message = "You are an AI assistant with access to various tools. "
            if selected_tools:
                tool_descriptions = []
                for tool_name in selected_tools:
                    if tool_name in AVAILABLE_TOOLS:
                        tool = AVAILABLE_TOOLS[tool_name]
                        tool_descriptions.append(f"- {tool.name}: {tool.description}")

                system_message += f"You have access to these tools:\n" + "\n".join(tool_descriptions)
                system_message += "\n\nWhen a user asks something that could benefit from using these tools, suggest which tool to use and I'll execute it for you."

            # Prepare messages for Ollama
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            # Send streaming response
            await manager.send_personal_message({
                "type": "assistant_response_start",
                "timestamp": datetime.now().isoformat()
            }, client_id)

            full_response = ""
            async for chunk in stream_ollama_response(messages, model):
                full_response += chunk
                await manager.send_personal_message({
                    "type": "assistant_response_chunk",
                    "content": chunk,
                    "timestamp": datetime.now().isoformat()
                }, client_id)

            # Send assistant response complete BEFORE tool execution
            await manager.send_personal_message({
                "type": "assistant_response_complete",
                "timestamp": datetime.now().isoformat()
            }, client_id)

            # If tools are selected, use multi-agent system for intelligent orchestration
            if selected_tools:
                await manager.send_personal_message({
                    "type": "tool_execution_start",
                    "timestamp": datetime.now().isoformat()
                }, client_id)

                try:
                    # Get multi-agent system for the selected model
                    mas = get_multi_agent_system(model)

                    # Execute query using multi-agent system
                    mas_result = mas.execute_query(user_message, selected_tools, attached_file)

                    if mas_result.get("success"):
                        # Send orchestrator's final answer
                        await manager.send_personal_message({
                            "type": "agent_response",
                            "content": mas_result.get("final_answer", ""),
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                        # Send results from each sub-agent
                        for agent_result in mas_result.get("agent_results", []):
                            if agent_result.get("success"):
                                tool_name = agent_result.get("tool", "unknown")

                                await manager.send_personal_message({
                                    "type": "tool_result",
                                    "tool": tool_name,
                                    "result": agent_result.get("result", {}),
                                    "timestamp": datetime.now().isoformat()
                                }, client_id)

                                # Generate summary of tool result
                                base_summary = get_ollama_summary(tool_name, agent_result.get("result", {}), model)

                                # Include file summary if available from data processing
                                if tool_name == "data_processing" and agent_result.get("file_summary"):
                                    summary = f"File Summary:\n{agent_result.get('file_summary')}\n\nProcessing Result:\n{base_summary}"
                                else:
                                    summary = base_summary

                                await manager.send_personal_message({
                                    "type": "tool_summary",
                                    "tool": tool_name,
                                    "summary": summary,
                                    "timestamp": datetime.now().isoformat()
                                }, client_id)
                    else:
                        # Send error message
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Multi-agent execution failed: {mas_result.get('error', 'Unknown error')}",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Multi-agent execution error: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }, client_id)

                finally:
                    # Clean up temporary files
                    if attached_file and attached_file.get("path"):
                        try:
                            if os.path.exists(attached_file["path"]):
                                os.unlink(attached_file["path"])
                        except Exception as e:
                            print(f"Error cleaning up temporary file: {e}")

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        await manager.send_personal_message({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }, client_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)