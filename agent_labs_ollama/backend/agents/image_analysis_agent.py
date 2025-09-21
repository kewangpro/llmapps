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
            else:
                # Extract image path from query
                image_path = "image.jpg"  # Default
                for word in query.split():
                    if any(word.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
                        image_path = word
                        break
                file_path = image_path
                clean_query = query

            # Step 1: Call the image analysis tool to process the file (no LLM, just file I/O)
            tool_params = {"image_path": file_path}
            logger.info(f"🖼️ Calling image tool with: {tool_params}")

            tool_result = self._execute_tool_script("image_analysis", tool_params)

            if not tool_result.get("success", False):
                return {
                    "agent": "ImageAnalysisAgent",
                    "tool": "image_analysis",
                    "parameters": tool_params,
                    "result": tool_result,
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }

            # Step 2: Use LLM to analyze the image content from the tool's data
            logger.info(f"🖼️ Tool successful, now using LLM to analyze image content")

            # Determine analysis type from clean query
            analysis_type = "comprehensive"  # Default
            if "text" in clean_query.lower() or "ocr" in clean_query.lower():
                analysis_type = "text"
            elif "metadata" in clean_query.lower() or "exif" in clean_query.lower():
                analysis_type = "metadata"
            elif "basic" in clean_query.lower():
                analysis_type = "basic"

            # Prepare analysis prompt based on type
            analysis_prompts = {
                "comprehensive": """Analyze this image data and provide a detailed description including:
- Main subjects and objects visible
- Setting and environment
- Colors and composition
- Actions or activities
- Any text visible in the image
- Style and mood
- Notable details and features

Keep the description clear and comprehensive.""",

                "basic": """Provide a brief description of this image focusing on:
- What is the main subject?
- Where does this appear to be taken?
- What are the key visual elements?

Keep it concise but informative.""",

                "text": """Focus on extracting and describing any text visible in this image:
- All readable text (signs, labels, documents, etc.)
- Text location and context
- Language if not English
- Quality of text (clear, blurry, partial, etc.)

If no text is visible, clearly state that.""",

                "metadata": """Analyze the technical and contextual aspects of this image:
- Image quality and resolution
- File information and metadata
- Technical characteristics
- EXIF data if available

Focus on observable technical characteristics."""
            }

            analysis_prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])

            # Get image metadata from tool result
            file_info = tool_result.get("file_info", {})
            filename = file_info.get("filename", "image")
            file_size = file_info.get("file_size_mb", "unknown")
            dimensions = f"{file_info.get('width', 'unknown')}x{file_info.get('height', 'unknown')}"

            # Get the base64 image data for actual visual analysis
            image_data = tool_result.get("image_data", {})
            base64_data = image_data.get("base64", "")
            data_url = image_data.get("data_url", "")

            if not base64_data:
                logger.warning("🖼️ No base64 image data available, falling back to metadata analysis")
                # Create comprehensive prompt with image context
                full_prompt = f"""{analysis_prompt}

Image Context:
- Filename: {filename}
- File size: {file_size} MB
- Dimensions: {dimensions}
- Format: {file_info.get('format', 'unknown')}

Note: This analysis is based on the image file metadata and context only. The image contains visual content that would be analyzed if the image data was available."""

                # Use LLM to generate analysis
                ai_analysis = self.llm.call(full_prompt)
            else:
                logger.info("🖼️ Using base64 image data for visual content analysis")

                # Create prompt for actual image analysis with visual content
                visual_prompt = f"""{analysis_prompt}

Image Technical Details:
- Filename: {filename}
- File size: {file_size} MB
- Dimensions: {dimensions}
- Format: {file_info.get('format', 'unknown')}

Please analyze the actual visual content of this image and provide detailed insights."""

                # Use LLM with image data for visual analysis
                if self.llm.supports_vision():
                    logger.info("🖼️ LLM supports vision, performing visual content analysis")
                    ai_analysis = self.llm.call_with_image(visual_prompt, data_url)
                else:
                    # Fallback if LLM doesn't support image analysis
                    logger.warning("🖼️ LLM doesn't support image analysis, using text-only analysis")
                    fallback_prompt = f"""{visual_prompt}

[Image data is available ({file_size} MB, {dimensions}) but this LLM cannot process visual content. Analysis limited to technical metadata.]"""
                    ai_analysis = self.llm.call(fallback_prompt)

            # Combine tool result with AI analysis
            result = tool_result.copy()
            result["ai_analysis"] = ai_analysis
            result["analysis_type"] = analysis_type

            return {
                "agent": "ImageAnalysisAgent",
                "tool": "image_analysis",
                "parameters": {"image_path": file_path, "analysis_type": analysis_type},
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