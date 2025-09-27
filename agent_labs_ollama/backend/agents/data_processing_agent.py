"""
Data Processing Agent - Specialized agent for data processing operations
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("DataProcessingAgent")


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
            # Check for attached file path in query (added from orchestrator)
            if "FILE_PATH:" in query:
                file_path = query.split("FILE_PATH:")[-1].strip()
                # Remove the FILE_PATH marker from the query for operation detection
                clean_query = query.split("FILE_PATH:")[0].strip()
                logger.info(f"📎 Found attached file: {file_path}")
                
                # Read the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    logger.info(f"📄 Read file content: {len(file_content)} characters")
                    
                    # Determine operation from clean query
                    prompt = f"""The user has attached a file and wants to: "{clean_query}"

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
                    
                    # Use keyword-based detection instead of LLM for faster processing
                    operation = self._detect_operation_from_query(clean_query)
                    logger.info(f"📊 Detected operation: {operation}")

                    # Execute data processing tool directly (no chunking for better performance)
                    params = {"input_data": file_content, "operation": operation}
                    result = self._execute_tool_script("data_processing", params)

                    if not result.get("success", False):
                        return {
                            "tool": "data_processing",
                            "success": False,
                            "error": f"Data processing tool failed: {result.get('error', 'Unknown error')}",
                            "timestamp": datetime.now().isoformat()
                        }

                    # Skip LLM analysis for simple transformation operations
                    if operation in ["csv_to_json", "json_to_csv", "remove_duplicates", "clean_text", "sort_lines", "extract_emails", "extract_urls"]:
                        llm_analysis = f"Successfully completed {operation} operation. File ready for download."
                    else:
                        # Use LLM analysis only for complex operations like text_analysis
                        llm_analysis = self._analyze_processing_results_with_llm(result, query, operation)

                    # Format for downstream agents
                    formatted_tool_data = self._format_tool_data(result, operation)

                    result_data = {
                        "tool_data": formatted_tool_data,  # Formatted data for chaining
                        "llm_analysis": llm_analysis,      # LLM insights
                        # Preserve all original tool result fields for frontend
                        **{k: v for k, v in result.items() if k not in ["tool", "success", "timestamp"]}
                    }

                    response = {
                            "tool": "data_processing",
                        "parameters": params,
                        "result": result_data,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }

                    return response
                    
                except Exception as e:
                    logger.error(f"📄 Failed to read file {file_path}: {str(e)}")
                    return {
                        "tool": "data_processing",
                        "success": False,
                        "error": f"Failed to read file {file_path}: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Check if this appears to be a file attachment (contains file marker)
            elif "[Attached file:" in query:
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

            # Use keyword-based detection for non-file queries too
            operation = self._detect_operation_from_query(query)
            logger.info(f"📊 Detected operation: {operation}")

            # Execute data processing tool directly (no chunking)
            params = {"input_data": query, "operation": operation}
            result = self._execute_tool_script("data_processing", params)

            if not result.get("success", False):
                return {
                    "tool": "data_processing",
                    "success": False,
                    "error": f"Data processing tool failed: {result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # Skip LLM analysis for simple transformation operations
            if operation in ["csv_to_json", "json_to_csv", "remove_duplicates", "clean_text", "sort_lines", "extract_emails", "extract_urls"]:
                llm_analysis = f"Successfully completed {operation} operation. File ready for download."
            else:
                # Use LLM analysis only for complex operations like text_analysis
                llm_analysis = self._analyze_processing_results_with_llm(result, query, operation)

            # Format for downstream agents
            formatted_tool_data = self._format_tool_data(result, operation)

            result_data = {
                "tool_data": formatted_tool_data,  # Formatted data for chaining
                "llm_analysis": llm_analysis,      # LLM insights
                # Preserve all original tool result fields for frontend
                **{k: v for k, v in result.items() if k not in ["tool", "success", "timestamp"]}
            }

            # Add file summary if we generated one
            if "[Attached file:" in query and 'file_summary' in locals():
                result_data["file_summary"] = file_summary

            response = {
                "tool": "data_processing",
                "parameters": params,
                "result": result_data,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

            return response

        except Exception as e:
            return {
                "tool": "data_processing",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_processing_results_with_llm(self, tool_result: Dict[str, Any], original_query: str, operation: str) -> str:
        """Use LLM to analyze data processing results and provide insights"""
        try:
            analysis_prompt = f"""Analyze these data processing results and provide insights for the user query: "{original_query}"

Operation Performed: {operation}
Processing Results:
{self._format_results_for_analysis(tool_result)}

Please provide:
1. Summary of what was processed
2. Key findings and metrics
3. Insights about the data
4. Any recommendations or next steps if applicable

Format your response with:
- Clear section headers
- Important details highlighted
- Organized layout that's easy to read

Focus on the information most relevant to the user's question."""

            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"📄 Generated LLM analysis for data processing results")
            return llm_response.strip()

        except Exception as e:
            logger.error(f"📄 Error in LLM analysis: {str(e)}")
            return f"Data processing completed but LLM analysis failed: {str(e)}"

    def _format_results_for_analysis(self, result: Dict[str, Any]) -> str:
        """Format processing results for LLM analysis"""
        try:
            if not result.get("success", False):
                return f"Processing failed: {result.get('error', 'Unknown error')}"

            operation = result.get("operation", "unknown")
            output = result.get("output", {})
            message = result.get("message", "")

            formatted = f"Operation: {operation}\n"
            if message:
                formatted += f"Summary: {message}\n"

            if isinstance(output, dict):
                formatted += "Results:\n"
                for key, value in output.items():
                    formatted += f"  {key}: {value}\n"
            else:
                formatted += f"Output: {str(output)}\n"

            return formatted
        except Exception as e:
            logger.error(f"📄 Error formatting results: {str(e)}")
            return "Error formatting processing results"

    def _format_tool_data(self, tool_result: Dict[str, Any], operation: str) -> str:
        """Format tool result as text for downstream agents"""
        try:
            if not tool_result.get("success", False):
                return f"Processing failed for operation '{operation}': {tool_result.get('error', 'Unknown error')}"

            output = tool_result.get("output", {})
            message = tool_result.get("message", "")

            formatted_text = f"Data Processing Operation: {operation}\n"

            if message:
                formatted_text += f"Summary: {message}\n"

            if isinstance(output, dict):
                formatted_text += "Results:\n"
                for key, value in output.items():
                    formatted_text += f"- {key}: {value}\n"
            elif isinstance(output, list):
                formatted_text += f"Results: {len(output)} items\n"
                # Show first few items if it's a list
                for i, item in enumerate(output[:5]):
                    formatted_text += f"- {item}\n"
                if len(output) > 5:
                    formatted_text += f"... and {len(output) - 5} more items\n"
            else:
                formatted_text += f"Result: {str(output)}\n"

            return formatted_text
        except Exception as e:
            logger.error(f"📄 Error formatting tool data: {str(e)}")
            return f"Error formatting data processing results: {str(e)}"

    def _detect_operation_from_query(self, query: str) -> str:
        """Detect operation from query using keywords (no LLM needed)"""
        query_lower = query.lower()

        # Conversion operations
        if "csv to json" in query_lower or "convert csv" in query_lower:
            return "csv_to_json"
        elif "json to csv" in query_lower or "convert json" in query_lower:
            return "json_to_csv"

        # Extraction operations
        elif "email" in query_lower and ("extract" in query_lower or "get" in query_lower or "find" in query_lower):
            return "extract_emails"
        elif "url" in query_lower and ("extract" in query_lower or "get" in query_lower or "find" in query_lower):
            return "extract_urls"

        # Cleaning operations
        elif "remove duplicates" in query_lower or "deduplicate" in query_lower:
            return "remove_duplicates"
        elif "clean" in query_lower:
            return "clean_text"
        elif "sort" in query_lower:
            return "sort_lines"

        # Analysis operations
        elif "analyze" in query_lower or "analysis" in query_lower:
            return "text_analysis"
        elif "word count" in query_lower or "count words" in query_lower:
            return "word_count"
        elif "statistics" in query_lower or "stats" in query_lower:
            return "calculate_stats"

        # Encoding operations
        elif "base64" in query_lower and "encode" in query_lower:
            return "base64_encode"
        elif "base64" in query_lower and "decode" in query_lower:
            return "base64_decode"
        elif "format json" in query_lower:
            return "json_format"

        # Default fallback
        else:
            return "text_analysis"