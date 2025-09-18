"""
File Search Agent - Specialized agent for file search operations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class FileSearchAgent(BaseAgent):
    """Specialized agent for file search operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute file search with intelligent parameter extraction"""
        try:
            logger.info(f"📁 FileSearchAgent analyzing: '{query}'")
            prompt = f"""Extract file search parameters from this query: "{query}"
Examples:
- "find images under the videos folder" → {{"pattern": "**/*.{{jpg,jpeg,png,gif,bmp,tiff,webp,svg}}", "path": "videos"}}
- "find Python files in src" → {{"pattern": "**/*.py", "path": "src"}}
- "how many files under ~/videos folder?" → {{"pattern": "*", "path": "~/videos"}}
- "count files in documents" → {{"pattern": "*", "path": "documents"}}
- "search for config files" → {{"pattern": "*config*", "path": "."}}
- "find all text files" → {{"pattern": "**/*.txt", "path": "."}}
File type mappings:
- images: *.{{jpg,jpeg,png,gif,bmp,tiff,webp,svg}}
- videos: *.{{mp4,avi,mov,mkv,wmv,flv,webm}}
- documents: *.{{pdf,doc,docx,txt,rtf,odt}}
- code: *.{{py,js,html,css,java,cpp,c,h,rb,php,go,rs}}
- data: *.{{csv,json,xml,yaml,yml}}
Determine:
1. Search pattern - use "*" for counting all files, "**/*" for recursive, specific patterns for file types
2. Path - extract directory paths like "~/videos", "documents", "src", etc. Use "." for current directory
Respond with JSON only:
{{"pattern": "search_pattern", "path": "directory_path"}}
For counting queries, use pattern "*" or "**/*" for recursive."""
            response = self.llm.call(prompt)
            logger.info(f"📁 Parameter extraction: {response.strip()}")
            try:
                # Clean up the response - remove markdown formatting
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]  # Remove ```
                clean_response = clean_response.strip()
                params = json.loads(clean_response)
                logger.info(f"📁 Successfully parsed parameters: {params}")
            except json.JSONDecodeError:
                # Fallback to simple pattern
                params = {"pattern": f"*{query}*"}
                logger.info(f"📁 Using fallback parameters: {params}")

            logger.info(f"📁 Executing file_search tool with: {params}")
            # Execute file search tool
            result = self._execute_tool_script("file_search", params)
            logger.info(f"📁 Tool execution completed: {len(str(result))} characters")

            return {
                "agent": "FileSearchAgent",
                "tool": "file_search",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"📁 FileSearchAgent error: {str(e)}")
            return {
                "agent": "FileSearchAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }