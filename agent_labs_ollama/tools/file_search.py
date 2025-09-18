#!/usr/bin/env python3
"""
File Search Tool
Searches for files in the filesystem based on patterns
"""

import os
import glob
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

def search_files(path: str = ".", pattern: str = "*", max_results: int = 50) -> Dict[str, Any]:
    """
    Search for files matching a pattern in the given path

    Args:
        path: Directory path to search in (default: current directory)
        pattern: File pattern to search for (supports wildcards like *.py, *.txt)
        max_results: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    try:
        # Expand home directory if needed
        if path.startswith("~/"):
            path = os.path.expanduser(path)

        # Ensure path exists and is safe
        search_path = Path(path).resolve()
        if not search_path.exists():
            return {"error": f"Path does not exist: {path}"}

        if not search_path.is_dir():
            return {"error": f"Path is not a directory: {path}"}

        # Perform the search
        # Handle brace expansion patterns like *.{jpg,png,gif}
        if '{' in pattern and '}' in pattern:
            # Extract extensions from brace pattern
            import re
            brace_match = re.search(r'\*\.{([^}]+)}', pattern)
            if brace_match:
                extensions = brace_match.group(1).split(',')
                found_files = []
                for ext in extensions:
                    ext_pattern = pattern.replace('{' + brace_match.group(1) + '}', ext.strip())
                    search_pattern = str(search_path / ext_pattern)
                    found_files.extend(glob.glob(search_pattern, recursive=True))
                # Remove duplicates
                found_files = list(set(found_files))
            else:
                search_pattern = str(search_path / pattern)
                found_files = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = str(search_path / pattern)
            found_files = glob.glob(search_pattern, recursive=True)

        # Limit results
        found_files = found_files[:max_results]

        # Get file info
        results = []
        for file_path in found_files:
            file_info = {
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": os.path.getsize(file_path) if os.path.isfile(file_path) else 0,
                "is_file": os.path.isfile(file_path),
                "is_dir": os.path.isdir(file_path)
            }
            results.append(file_info)

        return {
            "tool": "file_search",
            "success": True,
            "query": {
                "path": path,
                "pattern": pattern
            },
            "results_count": len(results),
            "files": results,
            "message": f"Found {len(results)} items matching '{pattern}' in '{path}'"
        }

    except Exception as e:
        return {
            "tool": "file_search",
            "success": False,
            "error": str(e)
        }

def main():
    """CLI interface for the file search tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: file_search.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        path = args.get("path", ".")
        pattern = args.get("pattern", "*")
        max_results = args.get("max_results", 50)

        result = search_files(path, pattern, max_results)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()