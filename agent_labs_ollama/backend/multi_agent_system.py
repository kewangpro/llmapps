"""
Multi-Agent System for Agent Labs
Orchestrator pattern with specialized sub-agents for each tool
"""

import json
import os
import subprocess
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import httpx

# Configure logging for multi-agent system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiAgentSystem")


class OllamaLLM:
    """Simple Ollama LLM client"""

    def __init__(self, model: str = "gemma3:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def call(self, prompt: str) -> str:
        """Call Ollama API and return response"""
        try:
            with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
                response = client.post(
                    '/api/chat',
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    }
                )
                result = response.json()
                return result.get('message', {}).get('content', '')
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, model: str = "gemma3:latest"):
        self.model = model
        self.llm = OllamaLLM(model=model)
        # Since we run from project root, tools are in ./tools
        self.tools_dir = "tools"

    def _execute_tool_script(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool script with given parameters"""
        try:
            # Change to project root directory for tool execution (so .env can be found)
            project_root = os.path.join(os.path.dirname(__file__), "..")

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
                timeout=30,  # 30 second timeout
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


class FileSearchAgent(BaseAgent):
    """Specialized agent for file search operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute file search with intelligent parameter extraction"""
        try:
            logger.info(f"📁 FileSearchAgent analyzing: '{query}'")

            prompt = f"""Extract file search parameters from this query: "{query}"

Determine:
1. Search pattern (e.g., "*.py", "config", "README*")
2. Optional path (if mentioned)

Respond with JSON only:
{{"pattern": "search_pattern", "path": "optional_path"}}

If no specific pattern mentioned, use "*{query}*" for general search."""

            response = self.llm.call(prompt)
            logger.info(f"📁 Parameter extraction: {response.strip()}")

            try:
                params = json.loads(response.strip())
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


class WebSearchAgent(BaseAgent):
    """Specialized agent for web search operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute web search with query optimization"""
        try:
            logger.info(f"🌐 WebSearchAgent analyzing: '{query}'")

            prompt = f"""Optimize this search query for web search: "{query}"

Create a clear, focused search query that will get the best results.
Respond with just the optimized query, no additional text."""

            optimized_query = self.llm.call(prompt).strip()

            # Remove any surrounding quotes that the LLM might have added
            if optimized_query.startswith('"') and optimized_query.endswith('"'):
                optimized_query = optimized_query[1:-1]
            if optimized_query.startswith("'") and optimized_query.endswith("'"):
                optimized_query = optimized_query[1:-1]

            logger.info(f"🌐 Query optimization: '{query}' → '{optimized_query}'")

            # Use original query if optimization fails
            if not optimized_query or len(optimized_query) > 200:
                optimized_query = query
                logger.info(f"🌐 Using original query: '{optimized_query}'")

            params = {"query": optimized_query}
            logger.info(f"🌐 Executing web_search tool with: {optimized_query}")

            # Execute web search tool
            result = self._execute_tool_script("web_search", params)
            logger.info(f"🌐 Web search completed: {len(str(result))} characters")

            return {
                "agent": "WebSearchAgent",
                "tool": "web_search",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"🌐 WebSearchAgent error: {str(e)}")
            return {
                "agent": "WebSearchAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class SystemInfoAgent(BaseAgent):
    """Specialized agent for system information operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute system info with intelligent metric selection"""
        try:
            logger.info(f"💻 SystemInfoAgent analyzing: '{query}'")

            prompt = f"""Determine what system metric to check for: "{query}"

Choose the best metric:
- "overview" for general system information
- "cpu" for CPU usage and details
- "memory" for RAM usage
- "disk" for storage information
- "network" for network details

Respond with just the metric name."""

            metric = self.llm.call(prompt).strip().lower()
            logger.info(f"💻 Selected metric: '{metric}'")

            # Validate metric
            valid_metrics = ["overview", "cpu", "memory", "disk", "network"]
            if metric not in valid_metrics:
                metric = "overview"
                logger.info(f"💻 Invalid metric, defaulting to: '{metric}'")

            params = {"metric": metric}
            logger.info(f"💻 Executing system_info tool with metric: {metric}")

            # Execute system info tool
            result = self._execute_tool_script("system_info", params)
            logger.info(f"💻 System info retrieved: {len(str(result))} characters")

            return {
                "agent": "SystemInfoAgent",
                "tool": "system_info",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"💻 SystemInfoAgent error: {str(e)}")
            return {
                "agent": "SystemInfoAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class CodeAnalysisAgent(BaseAgent):
    """Specialized agent for code analysis operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute code analysis with intelligent parameter extraction"""
        try:
            prompt = f"""Extract code analysis parameters from: "{query}"

Determine:
1. File path (if mentioned, otherwise use ".")
2. Analysis type: "general", "security", "performance", or "style"

Respond with JSON only:
{{"file_path": "path", "analysis_type": "type"}}"""

            response = self.llm.call(prompt)

            try:
                params = json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback parameters
                params = {"file_path": ".", "analysis_type": "general"}

            # Execute code analysis tool
            result = self._execute_tool_script("code_analysis", params)

            return {
                "agent": "CodeAnalysisAgent",
                "tool": "code_analysis",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "agent": "CodeAnalysisAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class DataProcessingAgent(BaseAgent):
    """Specialized agent for data processing operations"""

    def _generate_file_summary(self, file_content: str) -> str:
        """Generate file summary with chunking for large files"""
        # Define chunk size (characters) - adjust based on model context window
        chunk_size = 8000  # Conservative size for most models

        if len(file_content) <= chunk_size:
            # Small file - process directly
            summary_prompt = f"""Analyze and summarize this file content:

{file_content}

Provide a concise summary that includes:
1. File type and structure
2. Key contents/data
3. Notable patterns or important information
4. Potential security concerns if any

Keep the summary brief but informative."""

            return self.llm.call(summary_prompt)

        else:
            # Large file - chunk and process
            chunks = []
            for i in range(0, len(file_content), chunk_size):
                chunk = file_content[i:i + chunk_size]
                chunks.append(chunk)

            logger.info(f"📄 Large file detected ({len(file_content)} chars), processing in {len(chunks)} chunks")

            # Process each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"📄 Processing chunk {i}/{len(chunks)}")

                chunk_prompt = f"""Analyze this section (part {i} of {len(chunks)}) of a larger file:

{chunk}

Provide a brief summary focusing on:
1. Key information in this section
2. Important data or patterns
3. Any notable elements

Keep it concise as this is part of a larger analysis."""

                try:
                    chunk_summary = self.llm.call(chunk_prompt)
                    chunk_summaries.append(f"Section {i}: {chunk_summary}")
                except Exception as e:
                    logger.error(f"📄 Failed to process chunk {i}: {str(e)}")
                    chunk_summaries.append(f"Section {i}: [Processing failed - {str(e)[:100]}]")

            # Combine chunk summaries into final summary
            combined_content = "\n\n".join(chunk_summaries)

            final_prompt = f"""Based on these section summaries of a large file, create a comprehensive file summary:

{combined_content}

Provide a unified summary that includes:
1. Overall file type and structure
2. Main contents and key themes
3. Important patterns across all sections
4. Any security concerns or notable information

Keep the summary coherent and informative."""

            logger.info(f"📄 Generating final summary from {len(chunks)} processed chunks")
            return self.llm.call(final_prompt)

    def _process_large_input(self, query: str, operation: str) -> Dict[str, Any]:
        """Process large input data in chunks for text-based operations"""
        chunk_size = 8000

        # Extract file content if it's an attached file
        if "[Attached file:" in query:
            lines = query.split('\n')
            file_content_start = -1
            for i, line in enumerate(lines):
                if line.startswith('[Attached file:'):
                    file_content_start = i + 1
                    break

            if file_content_start > 0:
                content_to_process = '\n'.join(lines[file_content_start:])
            else:
                content_to_process = query
        else:
            content_to_process = query

        # Split into chunks
        chunks = []
        for i in range(0, len(content_to_process), chunk_size):
            chunk = content_to_process[i:i + chunk_size]
            chunks.append(chunk)

        logger.info(f"📄 Processing {operation} in {len(chunks)} chunks")

        # Process each chunk and aggregate results
        chunk_results = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"📄 Processing chunk {i}/{len(chunks)} for {operation}")

            params = {"input_data": chunk, "operation": operation}
            try:
                chunk_result = self._execute_tool_script("data_processing", params)
                chunk_results.append(chunk_result)
            except Exception as e:
                logger.error(f"📄 Failed to process chunk {i}: {str(e)}")
                chunk_results.append({
                    "tool": "data_processing",
                    "success": False,
                    "error": f"Chunk {i} failed: {str(e)[:100]}"
                })

        # Aggregate results based on operation type
        return self._aggregate_chunked_results(chunk_results, operation)

    def _aggregate_chunked_results(self, chunk_results: List[Dict], operation: str) -> Dict[str, Any]:
        """Aggregate results from chunked processing"""
        successful_results = [r for r in chunk_results if r.get("success", False)]

        if not successful_results:
            return {
                "tool": "data_processing",
                "success": False,
                "error": "All chunks failed to process"
            }

        if operation == "text_analysis":
            # Combine text analysis results
            total_chars = sum(r.get("output", {}).get("character_count", 0) for r in successful_results)
            total_words = sum(r.get("output", {}).get("word_count", 0) for r in successful_results)
            total_lines = sum(r.get("output", {}).get("line_count", 0) for r in successful_results)

            return {
                "tool": "data_processing",
                "success": True,
                "operation": operation,
                "output": {
                    "character_count": total_chars,
                    "word_count": total_words,
                    "line_count": total_lines,
                    "chunks_processed": len(successful_results)
                },
                "message": f"Analyzed text in {len(successful_results)} chunks: {total_words} words, {total_chars} characters"
            }

        elif operation == "word_count":
            # Combine word counts
            total_words = sum(r.get("output", {}).get("total_words", 0) for r in successful_results)
            unique_words = set()
            for r in successful_results:
                # This is simplified - actual unique word counting across chunks is more complex
                unique_words.update(str(r.get("output", {}).get("unique_words", 0)))

            return {
                "tool": "data_processing",
                "success": True,
                "operation": operation,
                "output": {
                    "total_words": total_words,
                    "chunks_processed": len(successful_results)
                },
                "message": f"Counted words across {len(successful_results)} chunks: {total_words} total words"
            }

        elif operation in ["extract_emails", "extract_urls"]:
            # Combine extracted items
            all_items = []
            for r in successful_results:
                items = r.get("output", [])
                if isinstance(items, list):
                    all_items.extend(items)

            unique_items = list(set(all_items))

            return {
                "tool": "data_processing",
                "success": True,
                "operation": operation,
                "output": unique_items,
                "message": f"Extracted {len(unique_items)} unique items from {len(successful_results)} chunks"
            }

        else:
            # For other operations, return the first successful result with chunk info
            first_result = successful_results[0]
            first_result["message"] = f"Processed in {len(successful_results)} chunks. Showing result from first chunk."
            return first_result

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute data processing with intelligent operation selection"""
        try:
            # Check if this appears to be a file attachment (contains file marker)
            if "[Attached file:" in query:
                # Extract file content for summarization
                lines = query.split('\n')
                file_content_start = -1

                for i, line in enumerate(lines):
                    if line.startswith('[Attached file:'):
                        file_content_start = i + 1
                        break

                if file_content_start > 0 and file_content_start < len(lines):
                    file_content = '\n'.join(lines[file_content_start:])
                    user_request = '\n'.join(lines[:file_content_start-1])

                    # Check file size and chunk if necessary
                    file_summary = self._generate_file_summary(file_content)

                    # Then determine the operation based on user request
                    prompt = f"""The user has attached a file and wants to: "{user_request}"

File summary: {file_summary}

Available operations:
- "json_format" - Format and validate JSON
- "csv_to_json" - Convert CSV to JSON
- "json_to_csv" - Convert JSON to CSV
- "text_analysis" - Analyze text metrics (word count, character count, etc.)
- "base64_encode" - Encode data to base64
- "base64_decode" - Decode base64 data
- "word_count" - Count words and unique words
- "extract_emails" - Extract email addresses from text
- "extract_urls" - Extract URLs from text
- "clean_text" - Clean and normalize text
- "sort_lines" - Sort lines alphabetically
- "remove_duplicates" - Remove duplicate lines
- "calculate_stats" - Calculate statistics for numeric data

Respond with just the operation name that best matches the request."""
                else:
                    prompt = f"""Determine the best data processing operation for: "{query}"

Available operations:
- "json_format" - Format and validate JSON
- "csv_to_json" - Convert CSV to JSON
- "json_to_csv" - Convert JSON to CSV
- "text_analysis" - Analyze text metrics (word count, character count, etc.)
- "base64_encode" - Encode data to base64
- "base64_decode" - Decode base64 data
- "word_count" - Count words and unique words
- "extract_emails" - Extract email addresses from text
- "extract_urls" - Extract URLs from text
- "clean_text" - Clean and normalize text
- "sort_lines" - Sort lines alphabetically
- "remove_duplicates" - Remove duplicate lines
- "calculate_stats" - Calculate statistics for numeric data

Respond with just the operation name that best matches the request."""
            else:
                prompt = f"""Determine the best data processing operation for: "{query}"

Available operations:
- "json_format" - Format and validate JSON
- "csv_to_json" - Convert CSV to JSON
- "json_to_csv" - Convert JSON to CSV
- "text_analysis" - Analyze text metrics (word count, character count, etc.)
- "base64_encode" - Encode data to base64
- "base64_decode" - Decode base64 data
- "word_count" - Count words and unique words
- "extract_emails" - Extract email addresses from text
- "extract_urls" - Extract URLs from text
- "clean_text" - Clean and normalize text
- "sort_lines" - Sort lines alphabetically
- "remove_duplicates" - Remove duplicate lines
- "calculate_stats" - Calculate statistics for numeric data

Respond with just the operation name that best matches the request."""

            operation = self.llm.call(prompt).strip().lower()

            # Validate operation
            valid_ops = ["json_format", "csv_to_json", "json_to_csv", "text_analysis",
                        "base64_encode", "base64_decode", "word_count", "extract_emails",
                        "extract_urls", "clean_text", "sort_lines", "remove_duplicates", "calculate_stats"]

            if operation not in valid_ops:
                operation = "text_analysis"  # Default to text analysis

            # Handle large input data by chunking for certain operations
            if len(query) > 10000 and operation in ["text_analysis", "word_count", "clean_text", "extract_emails", "extract_urls"]:
                logger.info(f"📄 Large input detected for {operation}, using chunked processing")
                result = self._process_large_input(query, operation)
            else:
                params = {"input_data": query, "operation": operation}
                # Execute data processing tool
                result = self._execute_tool_script("data_processing", params)

            # Include file summary in response if available
            response = {
                "agent": "DataProcessingAgent",
                "tool": "data_processing",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

            # Add file summary if we generated one
            if "[Attached file:" in query and 'file_summary' in locals():
                response["file_summary"] = file_summary

            return response

        except Exception as e:
            return {
                "agent": "DataProcessingAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class PresentationAgent(BaseAgent):
    """Specialized agent for generating PowerPoint presentations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute presentation generation with intelligent parameter extraction"""
        try:
            logger.info(f"🎨 PresentationAgent analyzing: '{query}'")

            # Extract parameters from query using LLM
            prompt = f"""Extract presentation generation parameters from: "{query}"

Determine:
1. Input text content (the main content to convert to slides)
2. Presentation title (if mentioned, otherwise generate one)
3. Output filename (if mentioned, otherwise use default)

Respond with JSON only:
{{"input_text": "content", "title": "title", "output_filename": "filename.pptx"}}

If the query contains file content (after [Attached file:]), use that as input_text."""

            response = self.llm.call(prompt)
            logger.info(f"🎨 Parameter extraction: {response.strip()}")

            try:
                params = json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback parameters
                params = {"input_text": query, "title": "Generated Presentation", "output_filename": "presentation.pptx"}

            # Execute presentation generation tool
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


class ImageAnalysisAgent(BaseAgent):
    """Specialized agent for image analysis operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute image analysis with intelligent parameter extraction"""
        try:
            logger.info(f"🖼️ ImageAnalysisAgent analyzing: '{query}'")

            # Extract parameters from query using LLM
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

                params = {"image_path": image_path, "analysis_type": "comprehensive"}

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


class OrchestratorAgent(BaseAgent):
    """Main orchestrator agent that coordinates sub-agents"""

    def __init__(self, model: str = "gemma3:latest"):
        super().__init__(model)
        self.sub_agents = {
            "file_search": FileSearchAgent(model),
            "web_search": WebSearchAgent(model),
            "system_info": SystemInfoAgent(model),
            "code_analysis": CodeAnalysisAgent(model),
            "data_processing": DataProcessingAgent(model),
            "presentation": PresentationAgent(model),
            "image_analysis": ImageAnalysisAgent(model)
        }

    def _select_agents(self, query: str, available_tools: List[str]) -> List[str]:
        """Select which sub-agents to use based on query and available tools"""
        try:
            logger.info(f"\n🧠 ORCHESTRATOR ANALYZING QUERY: '{query}'")
            logger.info(f"📋 Available tools: {available_tools}")

            tools_desc = {
                "file_search": "search for files and directories",
                "web_search": "search the internet for information",
                "system_info": "get system information (CPU, memory, disk, network)",
                "code_analysis": "analyze code for quality, security, performance",
                "data_processing": "process, analyze, or transform data",
                "presentation": "generate PowerPoint presentations from text or files",
                "image_analysis": "analyze image files for content, text, and metadata"
            }

            available_desc = [f"- {tool}: {tools_desc[tool]}" for tool in available_tools if tool in tools_desc]

            prompt = f"""Given this user query: "{query}"

Available tools:
{chr(10).join(available_desc)}

Which tools should be used? Consider:
1. What information is the user asking for?
2. What actions need to be performed?
3. Are multiple tools needed?

Respond with just the tool names, one per line. If multiple tools needed, list them in execution order."""

            logger.info("🤔 Orchestrator thinking...")
            response = self.llm.call(prompt)
            logger.info(f"💭 Orchestrator reasoning: {response.strip()}")

            # Parse response and validate against available tools
            selected = []
            for line in response.strip().split('\n'):
                tool = line.strip().lower().replace('-', '').replace('*', '').strip()
                if tool in available_tools:
                    selected.append(tool)

            # Fallback: if no valid selection, use first available tool
            if not selected and available_tools:
                selected = [available_tools[0]]
                logger.info(f"⚠️  Fallback selection: {selected}")
            else:
                logger.info(f"✅ Selected agents: {selected}")

            return selected

        except Exception as e:
            logger.error(f"❌ Agent selection error: {str(e)}")
            # Fallback: use first available tool
            fallback = [available_tools[0]] if available_tools else []
            logger.info(f"🔄 Emergency fallback: {fallback}")
            return fallback

    def execute(self, query: str, available_tools: List[str] = None) -> Dict[str, Any]:
        """Orchestrate sub-agents to handle the user query"""
        try:
            logger.info(f"\n🎯 ORCHESTRATOR STARTING EXECUTION")
            logger.info(f"Query: '{query}'")

            if not available_tools:
                available_tools = list(self.sub_agents.keys())

            # Select appropriate sub-agents
            selected_agents = self._select_agents(query, available_tools)

            results = []
            final_answer = ""

            logger.info(f"\n🚀 EXECUTING {len(selected_agents)} SUB-AGENTS...")

            # Execute selected sub-agents sequentially with context
            for i, agent_name in enumerate(selected_agents, 1):
                if agent_name in self.sub_agents:
                    logger.info(f"\n🤖 SUB-AGENT {i}/{len(selected_agents)}: {agent_name.upper()}Agent")

                    # Build context-aware query for agents after the first one
                    if i == 1:
                        # First agent uses original query
                        agent_query = query
                        logger.info(f"🎯 Delegating original query to {agent_name} specialist...")
                    else:
                        # Subsequent agents get context from previous results
                        logger.info(f"🔗 Building context-aware query for {agent_name} based on previous results...")

                        context = "Previous agent results:\n"
                        for j, prev_result in enumerate(results, 1):
                            if prev_result.get("success"):
                                context += f"{j}. {prev_result.get('agent', 'Unknown')}: {json.dumps(prev_result.get('result', {}), indent=2)}\n\n"

                        # Let the LLM create a contextual query for this agent
                        context_prompt = f"""Original user query: "{query}"

{context}

Based on the original query and the previous results above, create a specific query for the {agent_name} agent that takes advantage of the previous results.

For example:
- If system_info was run first and now we need web_search, use the specific OS version from system_info to search for updates
- If file_search found files and now we need code_analysis, focus on analyzing the specific files found

Respond with just the refined query, no additional text."""

                        agent_query = self.llm.call(context_prompt).strip()
                        logger.info(f"🎯 Context-aware query for {agent_name}: '{agent_query}'")

                    agent_result = self.sub_agents[agent_name].execute(agent_query)

                    if agent_result.get("success"):
                        logger.info(f"✅ {agent_name.upper()}Agent completed successfully")
                        logger.info(f"🔧 Tool used: {agent_result.get('tool', 'unknown')}")
                        logger.info(f"📊 Result preview: {str(agent_result.get('result', {}))[:100]}...")
                    else:
                        logger.error(f"❌ {agent_name.upper()}Agent failed: {agent_result.get('error', 'Unknown error')}")

                    results.append(agent_result)

            # Generate final answer based on all results
            if results:
                logger.info(f"\n🧠 SYNTHESIZING FINAL ANSWER...")
                logger.info(f"📝 Combining results from {len(results)} agents...")

                prompt = f"""Based on these tool execution results, provide a comprehensive answer to: "{query}"

Results:
{json.dumps(results, indent=2)}

Provide a clear, helpful response that synthesizes the information from the tools."""

                final_answer = self.llm.call(prompt)
                logger.info(f"✅ Final answer generated ({len(final_answer)} characters)")

            logger.info(f"\n🎉 ORCHESTRATION COMPLETE!")
            logger.info(f"📊 Total agents used: {len(selected_agents)}")
            logger.info(f"🎯 Successful executions: {sum(1 for r in results if r.get('success'))}")

            return {
                "orchestrator": "OrchestratorAgent",
                "query": query,
                "selected_agents": selected_agents,
                "agent_results": results,
                "final_answer": final_answer,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ ORCHESTRATION FAILED: {str(e)}")
            return {
                "orchestrator": "OrchestratorAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class MultiAgentSystem:
    """Main multi-agent system interface"""

    def __init__(self, model: str = "gemma3:latest"):
        self.orchestrator = OrchestratorAgent(model)

    def execute_query(self, query: str, selected_tools: List[str] = None) -> Dict[str, Any]:
        """Execute a query using the multi-agent system"""
        return self.orchestrator.execute(query, selected_tools)

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        return [
            {"name": "file_search", "description": "Search for files and directories in the filesystem"},
            {"name": "web_search", "description": "Search the web for current information and news"},
            {"name": "system_info", "description": "Get system information including CPU, memory, disk usage"},
            {"name": "code_analysis", "description": "Analyze code files for quality, security, and performance"},
            {"name": "data_processing", "description": "Process, analyze, and transform data"},
            {"name": "presentation", "description": "Generate PowerPoint presentations from text or files"},
            {"name": "image_analysis", "description": "Analyze image files for content, objects, text, and metadata"}
        ]