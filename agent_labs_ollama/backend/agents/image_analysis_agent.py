"""
Image Analysis Agent - Specialized agent for image analysis operations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("ImageAnalysisAgent")


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
                # Extract image path from query using pattern matching
                import re

                # Look for file paths that end with image extensions
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg']

                # Pattern to match file paths (both absolute and relative)
                # This will match paths like /path/to/file.jpg or ./docs/image.png or filename.jpg
                # Updated to better capture full paths including absolute paths starting with /
                pattern = r'(/[^\s]*\.(?:' + '|'.join(ext[1:] for ext in image_extensions) + r')|[./][^\s]*\.(?:' + '|'.join(ext[1:] for ext in image_extensions) + r')|[^\s]*\.(?:' + '|'.join(ext[1:] for ext in image_extensions) + r'))'
                matches = re.findall(pattern, query, re.IGNORECASE)

                if matches:
                    # Prioritize absolute paths (starting with /) over relative paths and filenames
                    absolute_paths = [m for m in matches if m.startswith('/')]
                    if absolute_paths:
                        file_path = absolute_paths[0]  # Use first absolute path
                    else:
                        file_path = matches[0]  # Use the first match as fallback
                    logger.info(f"🖼️ Extracted image path from query: {file_path}")
                    logger.info(f"🖼️ All matches found: {matches}")
                else:
                    # Fallback: try simple word splitting
                    file_path = "image.jpg"  # Default
                    for word in query.split():
                        if any(word.lower().endswith(ext) for ext in image_extensions):
                            file_path = word.rstrip('.,')  # Remove trailing punctuation
                            break
                    logger.info(f"🖼️ Using fallback extraction: {file_path}")

                clean_query = query

            # Step 1: Call the image analysis tool to process the file (no LLM, just file I/O)
            tool_params = {"image_path": file_path}
            logger.info(f"🖼️ Calling image tool with: {tool_params}")

            tool_result = self._execute_tool_script("image_analysis", tool_params)

            if not tool_result.get("success", False):
                error_msg = tool_result.get("error", "Image analysis tool failed")
                logger.error(f"🖼️ Tool failed: {error_msg}")
                return {
                    "tool": "image_analysis",
                    "parameters": tool_params,
                    "result": tool_result,
                    "success": False,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }

            # Step 2: Use LLM to analyze the image content from the tool's data
            logger.info(f"🖼️ Tool successful, now using LLM to analyze image content")

            # Use LLM to determine user intent and what they want
            intent_prompt = f"""Analyze this user request: "{clean_query}"

Determine what the user wants from the image analysis. Choose ONE of these response types:

1. "raw_exif" - User wants just the raw EXIF data without interpretation (e.g., "show EXIF data", "display EXIF", "get EXIF info")
2. "text" - User wants text extraction/OCR (e.g., "read text", "extract text", "what does it say")
3. "metadata" - User wants technical analysis including EXIF interpretation (e.g., "analyze metadata", "technical details")
4. "basic" - User wants simple description (e.g., "what is this", "describe briefly")
5. "comprehensive" - User wants detailed analysis (e.g., "analyze this image", "tell me about this")

Respond with just the type name (e.g., "raw_exif")."""

            try:
                analysis_type = self.llm.call(intent_prompt).strip().lower()
                # Validate response
                valid_types = ["raw_exif", "text", "metadata", "basic", "comprehensive"]
                if analysis_type not in valid_types:
                    analysis_type = "comprehensive"  # Default fallback
            except Exception:
                analysis_type = "comprehensive"  # Default fallback

            # Get image metadata from tool result (needed for all analysis types)
            file_info = tool_result.get("file_info", {})
            filename = file_info.get("filename", "image")
            file_size = file_info.get("file_size_mb", "unknown")
            dimensions = f"{file_info.get('width', 'unknown')}x{file_info.get('height', 'unknown')}"
            exif_data = file_info.get("exif", {})

            # Handle raw EXIF request - return data directly without LLM analysis
            if analysis_type == "raw_exif":
                if exif_data:
                    exif_output = "EXIF Data:\n"
                    for key, value in exif_data.items():
                        exif_output += f"{key}: {value}\n"
                else:
                    exif_output = "No EXIF data available in this image."

                result = {
                    "tool_data": exif_output,
                    "llm_analysis": exif_output,
                    "image_data": tool_result.get("image_data", {}),
                    "file_info": tool_result.get("file_info", {}),
                    "metadata": tool_result
                }

                return {
                    "tool": "image_analysis",
                    "parameters": {"image_path": file_path, "analysis_type": analysis_type},
                    "result": result,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }

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


            # Get the base64 image data for actual visual analysis
            image_data = tool_result.get("image_data", {})
            base64_data = image_data.get("base64", "")
            data_url = image_data.get("data_url", "")

            if not base64_data:
                logger.warning("🖼️ No base64 image data available, falling back to metadata analysis")
                # Create comprehensive prompt with image context
                exif_info = ""
                if exif_data:
                    exif_info = f"\nEXIF Data:\n"
                    for key, value in exif_data.items():
                        exif_info += f"- {key}: {value}\n"
                else:
                    exif_info = "\nEXIF Data: No EXIF data available in this image\n"

                full_prompt = f"""{analysis_prompt}

Image Context:
- Filename: {filename}
- File size: {file_size} MB
- Dimensions: {dimensions}
- Format: {file_info.get('format', 'unknown')}{exif_info}
Note: This analysis is based on the image file metadata and context only. The image contains visual content that would be analyzed if the image data was available."""

                # Use LLM to generate analysis
                ai_analysis = self.llm.call(full_prompt)
            else:
                logger.info("🖼️ Using base64 image data for visual content analysis")

                # Create prompt for actual image analysis with visual content
                exif_info = ""
                if exif_data:
                    exif_info = f"\nEXIF Data:\n"
                    for key, value in exif_data.items():
                        exif_info += f"- {key}: {value}\n"
                else:
                    exif_info = "\nEXIF Data: No EXIF data available in this image\n"

                visual_prompt = f"""{analysis_prompt}

Image Technical Details:
- Filename: {filename}
- File size: {file_size} MB
- Dimensions: {dimensions}
- Format: {file_info.get('format', 'unknown')}{exif_info}
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

            # Standardize result to match contract schema
            result = {
                "tool_data": ai_analysis,          # MUST: text analysis as tool_data for chaining
                "llm_analysis": ai_analysis,       # MUST: text analysis output
                "image_data": tool_result.get("image_data", {}),  # Image data for frontend display
                "file_info": tool_result.get("file_info", {}),   # File info for frontend display
                "metadata": tool_result            # Additional tool metadata
            }

            return {
                "tool": "image_analysis",
                "parameters": {"image_path": file_path, "analysis_type": analysis_type},
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"🖼️ ImageAnalysisAgent error: {str(e)}")
            return {
                "tool": "image_analysis",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }