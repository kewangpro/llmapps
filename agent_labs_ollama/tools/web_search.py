#!/usr/bin/env python3
"""
Web Search Tool
Performs web searches using Google Custom Search API
"""

import json
import sys
import os
import requests
from urllib.parse import quote_plus
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def web_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Search the web using Google Custom Search API

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    try:
        # Get API credentials from environment variables
        api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

        if not api_key or not engine_id:
            return {
                "tool": "web_search",
                "success": False,
                "error": "Google Search API credentials not found. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env file"
            }

        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            'key': api_key,
            'cx': engine_id,
            'q': query,
            'num': min(limit, 10)  # Google Custom Search allows max 10 results per request
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            results = []
            items = data.get('items', [])

            for item in items:
                results.append({
                    "title": item.get('title', ''),
                    "url": item.get('link', ''),
                    "snippet": item.get('snippet', ''),
                    "source": "Google Search"
                })

            if results:
                return {
                    "tool": "web_search",
                    "success": True,
                    "query": query,
                    "results_count": len(results),
                    "results": results,
                    "message": f"Found {len(results)} results for '{query}'"
                }
            else:
                return {
                    "tool": "web_search",
                    "success": True,
                    "query": query,
                    "results_count": 0,
                    "results": [],
                    "message": f"No results found for '{query}'"
                }

        elif response.status_code == 403:
            return {
                "tool": "web_search",
                "success": False,
                "error": "Google Custom Search API quota exceeded or invalid API key"
            }
        else:
            return {
                "tool": "web_search",
                "success": False,
                "error": f"Google Custom Search API error: {response.status_code} - {response.text}"
            }

    except requests.RequestException as e:
        return {
            "tool": "web_search",
            "success": False,
            "error": f"Network error during search: {str(e)}"
        }
    except Exception as e:
        return {
            "tool": "web_search",
            "success": False,
            "error": str(e)
        }

def main():
    """CLI interface for the web search tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: web_search.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        query = args.get("query", "")
        limit = args.get("limit", 5)

        if not query:
            print(json.dumps({"error": "query is required"}))
            sys.exit(1)

        result = web_search(query, limit)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()