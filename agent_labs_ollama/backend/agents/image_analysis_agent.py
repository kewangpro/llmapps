"""
Image Analysis Agent - Specialized agent for image analysis operations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class ImageAnalysisAgent(BaseAgent):
    """Specialized agent for image analysis operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute image analysis with intelligent parameter extraction"""
        try:
            logger.info(f"🖼️ ImageAnalysisAgent analyzing: '{query}'")

            # Check for attached file path in query
            if "FILE_PATH:" in query:
                file_path = query.split("FILE_PATH:")[-1].strip()
                # Remove the FILE_PATH marker from the query for analysis type detection
                clean_query = query.split("FILE_PATH:")[0].strip()
                logger.info(f"📎 Found attached file: {file_path}")

                # Determine analysis type from clean query
                analysis_type = "comprehensive"  # Default
                if "text" in clean_query.lower() or "ocr" in clean_query.lower():
                    analysis_type = "text"
                elif "metadata" in clean_query.lower() or "exif" in clean_query.lower():
                    analysis_type = "metadata"
                elif "basic" in clean_query.lower():
                    analysis_type = "basic"

                params = {"image_path": file_path, "analysis_type": analysis_type, "model": self.model}
                logger.info(f"🖼️ Using attached file parameters: {params}")
            else:
                # Extract parameters from query using LLM (fallback)
                prompt = f"""Extract image analysis parameters from: "{query}"

Determine:
1. Image path (look for file references or paths)
2. Analysis type: "comprehensive", "basic", "text", or "metadata"

Respond with JSON only:
{{"image_path": "path", "analysis_type": "type"}}

Analysis types:
- "comprehensive": Full analysis including metadata, text, and visual content
- "basic": Basic image properties and visual analysis
- "text": Focus on text extraction (OCR)
- "metadata": Focus on EXIF and metadata extraction"""

                response = self.llm.call(prompt)
                logger.info(f"🖼️ Parameter extraction: {response.strip()}")

                try:
                    params = json.loads(response.strip())
                except json.JSONDecodeError:
                    # Fallback parameters - look for image extensions in query
                    image_path = "image.jpg"  # Default
                    for word in query.split():
                        if any(word.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
                            image_path = word
                            break

                    params = {"image_path": image_path, "analysis_type": "comprehensive", "model": self.model}

            # Execute image analysis tool
            result = self._execute_tool_script("image_analysis", params)

            return {
                "agent": "ImageAnalysisAgent",
                "tool": "image_analysis",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"🖼️ ImageAnalysisAgent error: {str(e)}")
            return {
                "agent": "ImageAnalysisAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }