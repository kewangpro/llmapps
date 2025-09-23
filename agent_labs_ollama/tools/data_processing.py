#!/usr/bin/env python3
"""
Data Processing Tool
Processes and transforms data in various formats with downloadable outputs
"""

import json
import sys
import csv
import io
import re
import base64
from typing import Dict, Any, List, Union
from datetime import datetime

def process_data(input_data: str, operation: str) -> Dict[str, Any]:
    """
    Process data based on the specified operation

    Args:
        input_data: Input data to process (string, JSON, CSV, etc.)
        operation: Type of operation to perform

    Returns:
        Dictionary with processing results
    """
    try:
        operations = {
            "json_format": format_json,
            "csv_to_json": csv_to_json,
            "json_to_csv": json_to_csv,
            "text_analysis": analyze_text,
            "base64_encode": base64_encode,
            "base64_decode": base64_decode,
            "word_count": count_words,
            "extract_emails": extract_emails,
            "extract_urls": extract_urls,
            "clean_text": clean_text,
            "sort_lines": sort_lines,
            "remove_duplicates": remove_duplicates,
            "calculate_stats": calculate_stats
        }

        if operation not in operations:
            return {
                "tool": "data_processing",
                "success": False,
                "error": f"Unknown operation: {operation}. Available: {', '.join(operations.keys())}"
            }

        result = operations[operation](input_data)

        return {
            "tool": "data_processing",
            "success": True,
            "operation": operation,
            "input_length": len(input_data),
            "timestamp": datetime.now().isoformat(),
            **result
        }

    except Exception as e:
        return {
            "tool": "data_processing",
            "success": False,
            "error": str(e)
        }

def format_json(data: str) -> Dict[str, Any]:
    """Format and validate JSON"""
    try:
        parsed = json.loads(data)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        return {
            "output": formatted,
            "message": "JSON formatted successfully",
            "object_type": type(parsed).__name__,
            "size": len(formatted)
        }
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}

def csv_to_json(data: str) -> Dict[str, Any]:
    """Convert CSV to JSON with downloadable output"""
    try:
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        json_output = json.dumps(rows, indent=2, ensure_ascii=False)

        # Create downloadable file data
        filename = "converted_data.json"
        file_base64 = base64.b64encode(json_output.encode('utf-8')).decode('utf-8')
        file_size_mb = len(json_output.encode('utf-8')) / (1024 * 1024)

        return {
            "output": json_output,
            "message": f"Converted CSV to JSON with {len(rows)} rows",
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "file_size_mb": round(file_size_mb, 4),
            "processing_data": {
                "base64": file_base64,
                "filename": filename,
                "mime_type": "application/json",
                "content_preview": json_output[:300] + "..." if len(json_output) > 300 else json_output
            }
        }
    except Exception as e:
        return {"error": f"CSV conversion failed: {str(e)}"}

def json_to_csv(data: str) -> Dict[str, Any]:
    """Convert JSON to CSV with downloadable output"""
    try:
        parsed = json.loads(data)
        if not isinstance(parsed, list):
            return {"error": "JSON must be a list of objects for CSV conversion"}

        if not parsed:
            csv_output = ""
        else:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=parsed[0].keys())
            writer.writeheader()
            writer.writerows(parsed)
            csv_output = output.getvalue()

        # Create downloadable file data
        filename = "converted_data.csv"
        file_base64 = base64.b64encode(csv_output.encode('utf-8')).decode('utf-8')
        file_size_mb = len(csv_output.encode('utf-8')) / (1024 * 1024)

        return {
            "output": csv_output,
            "message": f"Converted JSON to CSV with {len(parsed)} rows",
            "row_count": len(parsed),
            "columns": list(parsed[0].keys()) if parsed else [],
            "file_size_mb": round(file_size_mb, 4),
            "processing_data": {
                "base64": file_base64,
                "filename": filename,
                "mime_type": "text/csv",
                "content_preview": csv_output[:300] + "..." if len(csv_output) > 300 else csv_output
            }
        }
    except Exception as e:
        return {"error": f"JSON to CSV conversion failed: {str(e)}"}

def analyze_text(data: str) -> Dict[str, Any]:
    """Analyze text for various metrics"""
    lines = data.split('\n')
    words = data.split()
    chars = len(data)
    chars_no_spaces = len(data.replace(' ', ''))

    return {
        "output": {
            "character_count": chars,
            "character_count_no_spaces": chars_no_spaces,
            "word_count": len(words),
            "line_count": len(lines),
            "sentence_count": len(re.findall(r'[.!?]+', data)),
            "paragraph_count": len([line for line in lines if line.strip()]),
            "average_word_length": round(sum(len(word) for word in words) / len(words), 2) if words else 0,
            "longest_word": max(words, key=len) if words else "",
            "most_common_words": get_word_frequency(data)
        },
        "message": f"Analyzed text with {len(words)} words and {chars} characters"
    }

def get_word_frequency(text: str) -> List[Dict[str, Union[str, int]]]:
    """Get word frequency analysis"""
    words = re.findall(r'\b\w+\b', text.lower())
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    # Return top 10 most common words
    sorted_words = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    return [{"word": word, "count": count} for word, count in sorted_words[:10]]

def base64_encode(data: str) -> Dict[str, Any]:
    """Encode data to base64"""
    encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
    return {
        "output": encoded,
        "message": f"Encoded {len(data)} characters to base64",
        "original_size": len(data),
        "encoded_size": len(encoded)
    }

def base64_decode(data: str) -> Dict[str, Any]:
    """Decode base64 data"""
    try:
        decoded = base64.b64decode(data).decode('utf-8')
        return {
            "output": decoded,
            "message": f"Decoded base64 to {len(decoded)} characters",
            "encoded_size": len(data),
            "decoded_size": len(decoded)
        }
    except Exception as e:
        return {"error": f"Base64 decode failed: {str(e)}"}

def count_words(data: str) -> Dict[str, Any]:
    """Count words in text"""
    words = data.split()
    unique_words = set(word.lower().strip('.,!?";') for word in words)

    return {
        "output": {
            "total_words": len(words),
            "unique_words": len(unique_words),
            "repetition_ratio": round((len(words) - len(unique_words)) / len(words), 2) if words else 0
        },
        "message": f"Found {len(words)} total words, {len(unique_words)} unique"
    }

def extract_emails(data: str) -> Dict[str, Any]:
    """Extract email addresses from text with downloadable output"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, data)
    unique_emails = list(set(emails))

    # Create output content
    output_content = "\n".join(unique_emails)

    # Create downloadable file data
    filename = "extracted_emails.txt"
    file_base64 = base64.b64encode(output_content.encode('utf-8')).decode('utf-8')
    file_size_mb = len(output_content.encode('utf-8')) / (1024 * 1024)

    return {
        "output": unique_emails,
        "message": f"Found {len(unique_emails)} unique email addresses",
        "total_matches": len(emails),
        "unique_matches": len(unique_emails),
        "file_size_mb": round(file_size_mb, 4),
        "processing_data": {
            "base64": file_base64,
            "filename": filename,
            "mime_type": "text/plain",
            "content_preview": output_content[:300] + "..." if len(output_content) > 300 else output_content
        }
    }

def extract_urls(data: str) -> Dict[str, Any]:
    """Extract URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, data)
    unique_urls = list(set(urls))

    return {
        "output": unique_urls,
        "message": f"Found {len(unique_urls)} unique URLs",
        "total_matches": len(urls),
        "unique_matches": len(unique_urls)
    }

def clean_text(data: str) -> Dict[str, Any]:
    """Clean and normalize text"""
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', data)
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    # Remove special characters (keep basic punctuation)
    cleaned = re.sub(r'[^\w\s.,!?;:-]', '', cleaned)

    return {
        "output": cleaned,
        "message": f"Cleaned text: {len(data)} → {len(cleaned)} characters",
        "original_length": len(data),
        "cleaned_length": len(cleaned),
        "reduction_percentage": round((len(data) - len(cleaned)) / len(data) * 100, 1) if data else 0
    }

def sort_lines(data: str) -> Dict[str, Any]:
    """Sort lines in text"""
    lines = data.split('\n')
    sorted_lines = sorted(lines)
    sorted_text = '\n'.join(sorted_lines)

    return {
        "output": sorted_text,
        "message": f"Sorted {len(lines)} lines alphabetically",
        "line_count": len(lines)
    }

def remove_duplicates(data: str) -> Dict[str, Any]:
    """Remove duplicate lines from text with downloadable output"""
    lines = data.split('\n')
    unique_lines = []
    seen = set()

    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    result_text = '\n'.join(unique_lines)

    # Create downloadable file data
    filename = "deduplicated_data.txt"
    file_base64 = base64.b64encode(result_text.encode('utf-8')).decode('utf-8')
    file_size_mb = len(result_text.encode('utf-8')) / (1024 * 1024)

    return {
        "output": result_text,
        "message": f"Removed duplicates: {len(lines)} → {len(unique_lines)} lines",
        "original_lines": len(lines),
        "unique_lines": len(unique_lines),
        "duplicates_removed": len(lines) - len(unique_lines),
        "file_size_mb": round(file_size_mb, 4),
        "processing_data": {
            "base64": file_base64,
            "filename": filename,
            "mime_type": "text/plain",
            "content_preview": result_text[:300] + "..." if len(result_text) > 300 else result_text
        }
    }

def calculate_stats(data: str) -> Dict[str, Any]:
    """Calculate statistics for numeric data"""
    try:
        # Try to extract numbers from the text
        numbers = re.findall(r'-?\d+\.?\d*', data)
        numbers = [float(num) for num in numbers]

        if not numbers:
            return {"error": "No numbers found in the input"}

        stats = {
            "count": len(numbers),
            "sum": sum(numbers),
            "average": round(sum(numbers) / len(numbers), 2),
            "minimum": min(numbers),
            "maximum": max(numbers),
            "range": max(numbers) - min(numbers)
        }

        # Calculate median
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        if n % 2 == 0:
            stats["median"] = round((sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2, 2)
        else:
            stats["median"] = sorted_numbers[n//2]

        return {
            "output": stats,
            "message": f"Calculated statistics for {len(numbers)} numbers",
            "numbers_found": numbers[:10]  # Show first 10 numbers
        }

    except Exception as e:
        return {"error": f"Statistics calculation failed: {str(e)}"}

def main():
    """CLI interface for the data processing tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: data_processing.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        input_data = args.get("input_data", "")
        operation = args.get("operation", "")

        if not input_data:
            print(json.dumps({"error": "input_data is required"}))
            sys.exit(1)

        if not operation:
            print(json.dumps({"error": "operation is required"}))
            sys.exit(1)

        result = process_data(input_data, operation)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()