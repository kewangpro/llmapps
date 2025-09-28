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

def get_multi_agent_system() -> MultiAgentSystem:
    """Get or create multi-agent system instance"""
    global multi_agent_system
    if multi_agent_system is None:
        logger.info("🚀 Initializing Multi-Agent System")
        multi_agent_system = MultiAgentSystem()
    return multi_agent_system

def update_model_selection(model: str, provider: str):
    """Update the model selection for the multi-agent system"""
    global multi_agent_system
    from llm_config import llm_config

    # Update the global LLM configuration
    llm_config.configure(provider, model)

    # Reset multi_agent_system to use new configuration
    multi_agent_system = None

# File handling utilities

def handle_attached_file(attached_file_data: Dict[str, Any], file_content: str) -> Optional[Dict[str, str]]:
    """Handle attached file by saving it to a temporary location"""
    if not attached_file_data or not file_content:
        return None

    try:
        file_name = attached_file_data.get("name", "unknown_file")
        file_type = attached_file_data.get("type", "")

        # Check if it's an image file sent as base64
        if file_type.startswith("image/") and file_content.startswith("data:"):
            logger.info(f"🔍 Processing image file: {file_name} ({file_type})")
            logger.info(f"🔍 Data URL preview: {file_content[:50]}...")

            # Extract base64 data from data URL
            if "," not in file_content:
                raise ValueError(f"Invalid data URL format: no comma separator found")

            header, base64_data = file_content.split(",", 1)
            logger.info(f"🔍 Base64 data length: {len(base64_data)}")


            # Decode base64 to binary with better error handling
            try:
                file_binary = base64.b64decode(base64_data, validate=True)
                logger.info(f"✅ Successfully decoded base64 data to {len(file_binary)} bytes")
            except Exception as decode_error:
                logger.error(f"❌ Base64 decode failed: {decode_error}")
                # Try alternative decoding without validation
                try:
                    file_binary = base64.b64decode(base64_data)
                    logger.info(f"✅ Alternative decode successful: {len(file_binary)} bytes")
                except Exception as alt_error:
                    raise ValueError(f"Failed to decode base64 data: {alt_error}")

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

# Note: get_tool_summary() function removed - OrchestratorAgent's final response serves as the summary

# Chat message model
class ChatMessage(BaseModel):
    content: str
    role: str = "user"
    tools: Optional[List[str]] = None

# API endpoints
@app.get("/api/tools")
async def get_tools():
    """Get available tools"""
    from multi_agent_system import MultiAgentSystem
    tools = MultiAgentSystem.get_available_tools()

    # Convert to the format expected by frontend
    tools_dict = {}
    for tool in tools:
        tool_id = tool["id"]
        tools_dict[tool_id] = {
            "id": tool_id,
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {},
            "category": tool.get("category", "general")
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
        full_model = model_data.get("model")
        if not full_model:
            return {"error": "Model parameter is required"}

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
            model = message_data.get("model")
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
                    raise ValueError(f"Invalid model format: '{model}'. Expected 'provider/model' format.")

                # Ensure LLM is configured with the selected model
                from llm_config import llm_config
                llm_config.configure(provider, model_name)

                mas = get_multi_agent_system()

                # Create a callback function for real-time messaging
                async def send_response_callback(response_type: str, content: str = "", **kwargs):
                    """Callback for the orchestrator to send real-time messages"""
                    if response_type == "initial_response":
                        # Send initial response immediately
                        await manager.send_personal_message({
                            "type": "assistant_response_start",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                        # Stream content preserving formatting (character-based)
                        chunk_size = 50
                        for i in range(0, len(content), chunk_size):
                            chunk = content[i:i + chunk_size]
                            await manager.send_personal_message({
                                "type": "assistant_response_chunk",
                                "content": chunk,
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

                        # Stream content preserving formatting (character-based)
                        chunk_size = 50
                        for i in range(0, len(content), chunk_size):
                            chunk = content[i:i + chunk_size]
                            await manager.send_personal_message({
                                "type": "assistant_response_chunk",
                                "content": chunk,
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
                # Send results for both successful and failed tools
                if mas_result.get("success"):
                    for agent_result in mas_result.get("agent_results", []):
                        tool_name = agent_result.get("tool", "unknown")

                        # Include error information for failed tools
                        result_data = agent_result.get("result", {})
                        if not agent_result.get("success", False) and agent_result.get("error"):
                            result_data = {"error": agent_result.get("error")}

                        await manager.send_personal_message({
                            "type": "tool_result",
                            "tool": tool_name,
                            "result": result_data,
                            "success": agent_result.get("success", False),
                            "timestamp": datetime.now().isoformat()
                        }, client_id)

                        # Note: Tool summaries removed - OrchestratorAgent's final response serves as the summary

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