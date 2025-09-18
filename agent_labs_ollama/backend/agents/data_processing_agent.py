"""
Data Processing Agent - Specialized agent for data processing operations
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


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