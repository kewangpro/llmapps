from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import httpx
from datetime import datetime
import base64
import tempfile
import os
import logging
from multi_agent_system import MultiAgentSystem

# Configure logging with timestamps
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AgentLabsServer")

app = FastAPI(title="Agent Labs", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://agent-labs-ollama-frontend-851143938786.us-central1.run.app"],
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
                websocket = self.active_connections[client_id]
                if websocket.client_state.name == 'CONNECTED':
                    # Add sequence number and flush immediately
                    message_json = json.dumps(message)
                    await websocket.send_text(message_json)
                    # Force flush the WebSocket
                    if hasattr(websocket, 'ping'):
                        try:
                            await websocket.ping()
                        except:
                            pass
                    logger.debug(f"✅ Successfully sent {message.get('type')} to {client_id}")
                else:
                    logger.warning(f"⚠️ WebSocket not connected (state: {websocket.client_state.name}) for {client_id}")
            except Exception as e:
                logger.error(f"❌ Failed to send message to {client_id}: {e}")
                # Remove the disconnected client
                self.disconnect(client_id)

manager = ConnectionManager()

import os

# Ollama client configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=30.0)

# Multi-Agent System management

# Initialize Multi-Agent System
multi_agent_system = None

def get_multi_agent_system(model: str = "gemma3:latest", provider: str = "ollama") -> MultiAgentSystem:
    """Get or create multi-agent system instance"""
    global multi_agent_system
    if multi_agent_system is None:
        logger.info(f"🚀 Initializing Multi-Agent System with {provider}/{model}")
        multi_agent_system = MultiAgentSystem(model=model, provider=provider)
    return multi_agent_system

def update_model_selection(model: str, provider: str):
    """Update the model selection for the multi-agent system"""
    global multi_agent_system
    from llm_config import llm_config

    # Update the global LLM configuration
    llm_config.configure(provider, model)

    # Reset multi_agent_system to use new configuration
    multi_agent_system = None

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
        logger.error(f"Error handling attached file: {e}")
        return None

def get_tool_summary(tool_name: str, tool_result: Dict[str, Any]) -> str:
    """Get a summary of tool results using the configured LLM"""
    try:
        from llm_config import llm_config

        prompt = f"""Please provide a concise, user-friendly summary of this tool execution result. Focus on the key information that would be useful to the user.

Tool: {tool_name}
Result: {json.dumps(tool_result, indent=2)}

Provide a brief, clear summary in 1-2 sentences highlighting the most important findings or data."""

        llm = llm_config.get_llm()
        return llm.call(prompt)

    except Exception as e:
        logger.error(f"Error generating tool summary: {e}")
        return f"Tool executed successfully but summary could not be generated: {str(e)}"

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

    # Define tool categories
    tool_categories = {
        # General Tools
        "file_search": "general",
        "web_search": "general",
        "system_info": "general",
        "presentation": "general",
        "visualization": "general",

        # Analytics Tools
        "cost_analysis": "analytics",
        "image_analysis": "analytics",
        "data_processing": "analytics",
        "stock_analysis": "analytics"
    }

    # Convert to the format expected by frontend
    tools_dict = {}
    for tool in tools:
        tool_name = tool["name"]
        tools_dict[tool_name] = {
            "name": tool_name,
            "description": tool["description"],
            "parameters": {},
            "category": tool_categories.get(tool_name, "general")
        }

    return {"tools": tools_dict}

@app.get("/api/models")
async def get_models():
    """Get available models from all providers"""
    from llm_config import llm_config

    # Get models from our configuration
    available_models = llm_config.get_available_models()

    models = []

    # Add OpenAI models
    for model_name in available_models.get("openai", {}):
        models.append({
            "name": f"openai/{model_name}",
            "provider": "openai",
            "model": model_name
        })

    # Add Gemini models
    for model_name in available_models.get("gemini", {}):
        models.append({
            "name": f"gemini/{model_name}",
            "provider": "gemini",
            "model": model_name
        })

    # Try to get Ollama models if available
    try:
        response = await ollama_client.get("/api/tags")
        ollama_data = response.json()
        for model in ollama_data.get("models", []):
            models.append({
                "name": f"ollama/{model['name']}",
                "provider": "ollama",
                "model": model['name']
            })
    except Exception as e:
        # Add default Ollama models if service unavailable
        for model_name in available_models.get("ollama", {}):
            models.append({
                "name": f"ollama/{model_name}",
                "provider": "ollama",
                "model": model_name
            })

    return {"models": models}


@app.post("/api/select-model")
async def select_model(model_data: dict):
    """Change the active model and provider"""
    try:
        full_model = model_data.get("model", "ollama/gemma3:latest")

        # Parse provider and model from format "provider/model"
        if "/" in full_model:
            provider, model = full_model.split("/", 1)
        else:
            # Default to ollama if no provider specified
            provider = "ollama"
            model = full_model

        # Update model selection
        update_model_selection(model, provider)

        return {
            "success": True,
            "provider": provider,
            "model": model,
            "message": f"Model changed to {provider}/{model}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


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
            logger.debug(f"Full message data keys: {list(message_data.keys())}")
            logger.info(f"Message: '{user_message}'")
            logger.info(f"Tools: {selected_tools}")
            logger.info(f"Model: {model}")
            logger.debug(f"Received attachedFile data: {attached_file_data}")
            logger.debug(f"AttachedFile type: {type(attached_file_data)}")

            if attached_file_data:
                logger.debug(f"File name: {attached_file_data.get('name')}")
                logger.debug(f"File type: {attached_file_data.get('type')}")
                logger.debug(f"File size: {attached_file_data.get('size')}")
                content = attached_file_data.get("content", "")
                logger.debug(f"Content length: {len(content) if content else 0}")
                logger.debug(f"Content preview: {content[:50]}..." if content else "No content")

            # Process attached file if present
            attached_file = None
            if attached_file_data and attached_file_data.get("content"):
                logger.info("Processing attached file...")
                attached_file = handle_attached_file(attached_file_data, attached_file_data.get("content"))
                logger.debug(f"Processed file result: {attached_file}")
            else:
                logger.debug("No attached file or content to process")

            # Send acknowledgment
            await manager.send_personal_message({
                "type": "message_received",
                "timestamp": datetime.now().isoformat()
            }, client_id)
            logger.info(f"📤 Sent message_received acknowledgment to client {client_id}")

            # Always use the multi-agent system with real-time communication
            try:
                logger.info(f"🤖 Processing message with Main Agent")
                # Parse provider and model from format "provider/model"
                if "/" in model:
                    provider, model_name = model.split("/", 1)
                else:
                    provider = "ollama"
                    model_name = model

                mas = get_multi_agent_system(model_name, provider)

                # Create a callback function for real-time messaging
                async def send_response_callback(response_type: str, content: str = "", **kwargs):
                    """Callback for the orchestrator to send real-time messages"""
                    if response_type == "initial_response":
                        # Send initial response immediately
                        await manager.send_personal_message({
                            "type": "assistant_response_start",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                        words = content.split()
                        current_chunk = ""
                        for word in words:
                            current_chunk += word + " "
                            if len(current_chunk) > 50:
                                await manager.send_personal_message({
                                    "type": "assistant_response_chunk",
                                    "content": current_chunk,
                                    "timestamp": datetime.now().isoformat()
                                }, client_id)
                                current_chunk = ""

                        if current_chunk:
                            await manager.send_personal_message({
                                "type": "assistant_response_chunk",
                                "content": current_chunk,
                                "timestamp": datetime.now().isoformat()
                            }, client_id)

                        await manager.send_personal_message({
                            "type": "assistant_response_complete",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                    elif response_type == "final_answer":
                        # Send final answer as separate response
                        await manager.send_personal_message({
                            "type": "assistant_response_start",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                        words = content.split()
                        current_chunk = ""
                        for word in words:
                            current_chunk += word + " "
                            if len(current_chunk) > 50:
                                await manager.send_personal_message({
                                    "type": "assistant_response_chunk",
                                    "content": current_chunk,
                                    "timestamp": datetime.now().isoformat()
                                }, client_id)
                                current_chunk = ""

                        if current_chunk:
                            await manager.send_personal_message({
                                "type": "assistant_response_chunk",
                                "content": current_chunk,
                                "timestamp": datetime.now().isoformat()
                            }, client_id)

                        await manager.send_personal_message({
                            "type": "assistant_response_complete",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                # Execute with callback - this will send responses in real-time
                mas_result = await mas.execute_query_with_callback(user_message, selected_tools, attached_file, send_response_callback)
                logger.info(f"🎯 Main Agent result: success={mas_result.get('success')}")

                # Send tool results if any (for transparency)
                # But responses are already sent via callback
                if mas_result.get("success"):
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
                            base_summary = get_tool_summary(tool_name, agent_result.get("result", {}))

                            await manager.send_personal_message({
                                "type": "tool_summary",
                                "tool": tool_name,
                                "summary": base_summary,
                                "timestamp": datetime.now().isoformat()
                            }, client_id)

                else:
                    # Send error message
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Main Agent error: {mas_result.get('error', 'Unknown error')}",
                        "timestamp": datetime.now().isoformat()
                    }, client_id)

            except Exception as e:
                logger.error(f"❌ Main Agent error: {str(e)}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Main Agent error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }, client_id)

            finally:
                # Clean up temporary files
                if attached_file and attached_file.get("path"):
                    try:
                        if os.path.exists(attached_file["path"]):
                            os.unlink(attached_file["path"])
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary file: {e}")

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