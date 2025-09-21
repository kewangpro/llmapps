"""
Base Agent class for all specialized agents
"""

import json
import os
import subprocess
import logging
from typing import Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod
import httpx
from llm_config import llm_config

logger = logging.getLogger("MultiAgentSystem")


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self):
        # Use shared LLM configuration
        self.llm = llm_config.get_llm()
        # Since we run from project root, tools are in ./tools
        self.tools_dir = "tools"

    def _execute_tool_script(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool script with given parameters"""
        try:
            # Change to project root directory for tool execution (so .env can be found)
            project_root = os.environ.get("APP_HOME")
            if not project_root:
                project_root = os.path.join(os.path.dirname(__file__), "..", "..")

            # Tool script path should be absolute
            tool_script = os.path.join(project_root, self.tools_dir, f"{tool_name}.py")

            if not os.path.exists(tool_script):
                return {"error": f"Tool script not found: {tool_script}"}

            # Prepare arguments as JSON
            args_json = json.dumps(parameters)

            # Execute the tool script using the virtual environment python (absolute path)
            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            if not os.path.exists(venv_python):
                venv_python = "python3"  # Fallback to system python

            result = subprocess.run(
                [venv_python, tool_script, args_json],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for image analysis
                cwd=project_root  # Run from project root so .env is accessible
            )

            if result.returncode == 0:
                try:
                    tool_output = json.loads(result.stdout)
                    return tool_output
                except json.JSONDecodeError:
                    return {
                        "tool": tool_name,
                        "success": False,
                        "error": "Tool returned invalid JSON",
                        "raw_output": result.stdout
                    }
            else:
                return {
                    "tool": tool_name,
                    "success": False,
                    "error": f"Tool execution failed with return code {result.returncode}",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }

        except subprocess.TimeoutExpired:
            return {
                "tool": tool_name,
                "success": False,
                "error": "Tool execution timed out (30 seconds)"
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "success": False,
                "error": f"Tool execution error: {str(e)}"
            }

    @abstractmethod
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the agent's task"""
        pass