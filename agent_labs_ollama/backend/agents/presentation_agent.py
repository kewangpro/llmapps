"""
Presentation Agent - Specialized agent for generating PowerPoint presentations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("PresentationAgent")


class PresentationAgent(BaseAgent):
    """Specialized agent for generating PowerPoint presentations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute presentation generation with intelligent parameter extraction"""
        try:
            logger.info(f"🎨 PresentationAgent analyzing: '{query}'")

            # Check for attached file path in query (similar to ImageAnalysisAgent)
            input_text = ""
            clean_query = query

            if "FILE_PATH:" in query:
                file_path = query.split("FILE_PATH:")[-1].strip()
                clean_query = query.split("FILE_PATH:")[0].strip()
                logger.info(f"📎 Found attached file: {file_path}")

                # Read the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        input_text = f.read()
                    logger.info(f"📄 Successfully read file content ({len(input_text)} characters)")
                except Exception as e:
                    logger.error(f"❌ Failed to read file {file_path}: {e}")
                    input_text = clean_query  # Fallback to query
            else:
                input_text = query

            # First, extract basic parameters from clean query using LLM
            param_prompt = f"""Extract presentation generation parameters from: "{clean_query}"

Determine:
1. Presentation title (if mentioned, otherwise generate one based on the content)
2. Output filename (if mentioned, otherwise use default)

Respond with JSON only:
{{"title": "title", "output_filename": "filename.pptx"}}

If no title is mentioned, generate an appropriate title."""

            param_response = self.llm.call(param_prompt)
            logger.info(f"🎨 Parameter extraction: {param_response.strip()}")

            try:
                basic_params = json.loads(param_response.strip())
            except json.JSONDecodeError:
                # Fallback parameters
                basic_params = {"title": "Generated Presentation", "output_filename": "presentation.pptx"}

            # Now use LLM to analyze input text and generate slide structure
            slide_prompt = f"""Analyze the following content and create a detailed PowerPoint presentation structure.

Content: {input_text}

Generate slides in this exact JSON format:
{{
    "slides_data": [
        {{"title": "Slide Title 1", "content": ["Bullet point 1", "Bullet point 2", "Bullet point 3"]}},
        {{"title": "Slide Title 2", "content": ["Bullet point 1", "Bullet point 2", "Bullet point 3"]}}
    ]
}}

Requirements:
- Create 5-8 slides minimum to cover the content comprehensively
- Each slide should have a clear title and 3-6 bullet points
- Focus on clear, concise bullet points
- Cover all major topics from the content
- Organize logically with introduction, main topics, and conclusion

Respond with only the JSON structure."""

            slide_response = self.llm.call(slide_prompt)
            logger.info(f"🎨 Slide generation: Generated slide structure")

            # Clean up the response to extract JSON
            cleaned_response = slide_response.strip()
            logger.info(f"🎨 Raw LLM response: {cleaned_response[:200]}...")

            # More robust JSON extraction
            import re
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()

            # Look for JSON structure in the response
            json_match = re.search(r'\{.*"slides_data".*\}', cleaned_response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group()

            logger.info(f"🎨 Cleaned JSON: {cleaned_response[:200]}...")

            try:
                slide_structure = json.loads(cleaned_response)
                slides_data = slide_structure.get("slides_data", [])
                if not slides_data:
                    raise ValueError("No slides_data found in response")
                logger.info(f"🎨 Successfully parsed {len(slides_data)} slides")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"🎨 Failed to parse slide structure: {e}")
                logger.error(f"🎨 LLM response was: {slide_response}")
                raise Exception(f"Unable to generate presentation structure from content. LLM parsing error: {e}")

            # Prepare parameters for the refactored tool
            params = {
                "slides_data": slides_data,
                "title": basic_params["title"],
                "output_filename": basic_params["output_filename"]
            }

            # Execute presentation creation tool
            result = self._execute_tool_script("presentation", params)

            return {
                "agent": "PresentationAgent",
                "tool": "presentation",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"🎨 PresentationAgent error: {str(e)}")
            return {
                "agent": "PresentationAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }