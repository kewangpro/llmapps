"""
File Search Agent - Specialized agent for file search operations
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("FileSearchAgent")


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
            tool_result = self._execute_tool_script("file_search", params)
            logger.info(f"📁 Tool execution completed: {len(str(tool_result))} characters")

            if not tool_result.get("success", False):
                return {
                    "agent": "FileSearchAgent",
                    "success": False,
                    "error": f"File search tool failed: {tool_result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Use LLM to analyze the search results
            llm_analysis = self._analyze_search_results_with_llm(tool_result, query)

            # Format for downstream agents
            formatted_tool_data = self._format_tool_data(tool_result)

            result = {
                "tool_data": formatted_tool_data,  # Formatted data for chaining
                "llm_analysis": llm_analysis       # LLM insights
            }

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

    def _analyze_search_results_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Use LLM to analyze file search results and provide insights"""
        try:
            # Extract relevant information from tool result
            files = tool_result.get("files", [])
            total_count = tool_result.get("total_count", len(files))
            search_path = tool_result.get("search_path", "")
            pattern = tool_result.get("pattern", "")

            analysis_prompt = f"""Analyze these file search results and provide insights for the user query: "{original_query}"

Search Pattern: {pattern}
Search Path: {search_path}
Total Files Found: {total_count}

File Results:
{self._format_files_for_analysis(files)}

Please provide:
1. Summary of what was found
2. File organization and structure insights
3. Key findings about the files (types, sizes, locations)
4. Any recommendations or next steps if applicable

Format your response with:
- Clear section headers
- Important details highlighted
- Organized layout that's easy to read

Focus on the information most relevant to the user's question."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"📁 Generated LLM analysis for file search results")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"📁 Error in LLM analysis: {str(e)}")
            return f"File search completed but LLM analysis failed: {str(e)}"

    def _format_files_for_analysis(self, files: list) -> str:
        """Format file list for LLM analysis"""
        try:
            if not files:
                return "No files found"

            formatted = ""
            for i, file_info in enumerate(files[:10], 1):  # Limit to top 10 files
                if isinstance(file_info, dict):
                    name = file_info.get("name", "Unknown")
                    path = file_info.get("path", "")
                    size = file_info.get("size", "")
                    modified = file_info.get("modified", "")

                    formatted += f"{i}. {name}\n"
                    if path:
                        formatted += f"   Path: {path}\n"
                    if size:
                        formatted += f"   Size: {size}\n"
                    if modified:
                        formatted += f"   Modified: {modified}\n"
                    formatted += "\n"
                else:
                    # Simple file path string
                    formatted += f"{i}. {file_info}\n"

            if len(files) > 10:
                formatted += f"... and {len(files) - 10} more files\n"

            return formatted
        except Exception as e:
            logger.error(f"📁 Error formatting files: {str(e)}")
            return "Error formatting file results"

    def _format_tool_data(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result as text for downstream agents"""
        try:
            files = tool_result.get("files", [])
            total_count = tool_result.get("total_count", len(files))
            search_path = tool_result.get("search_path", "")
            pattern = tool_result.get("pattern", "")

            if not files:
                return f"File search in '{search_path}' with pattern '{pattern}' found no files"

            formatted_text = f"File Search Results:\n"
            formatted_text += f"Search Path: {search_path}\n"
            formatted_text += f"Pattern: {pattern}\n"
            formatted_text += f"Total Files Found: {total_count}\n\n"

            # Show first few files
            formatted_text += "Files:\n"
            for i, file_info in enumerate(files[:5], 1):
                if isinstance(file_info, dict):
                    name = file_info.get("name", "Unknown")
                    path = file_info.get("path", "")
                    formatted_text += f"{i}. {name}"
                    if path:
                        formatted_text += f" ({path})"
                    formatted_text += "\n"
                else:
                    formatted_text += f"{i}. {file_info}\n"

            if len(files) > 5:
                formatted_text += f"... and {len(files) - 5} more files\n"

            return formatted_text
        except Exception as e:
            logger.error(f"📁 Error formatting tool data: {str(e)}")
            return f"Error formatting file search results: {str(e)}"