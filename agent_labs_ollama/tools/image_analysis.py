#!/usr/bin/env python3
"""
Image Analysis Tool
Analyzes image files using Ollama's vision capabilities
"""

import json
import sys
import os
from typing import Dict, Any
from datetime import datetime
import base64
import httpx

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 encoding"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to encode image: {str(e)}")

def analyze_image_with_ollama(image_path: str, analysis_type: str = "comprehensive", model: str = "gemma3:latest") -> Dict[str, Any]:
    """
    Analyze image using Ollama's vision model

    Args:
        image_path: Path to image file
        analysis_type: Type of analysis to perform
        model: Ollama vision model to use (default: llava)

    Returns:
        Dictionary with analysis results
    """
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            return {
                "tool": "image_analysis",
                "success": False,
                "error": f"Image file not found: {image_path}"
            }

        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)

        # Create analysis prompt based on type
        prompts = {
            "comprehensive": """Analyze this image comprehensively. Provide:
1. Main subject and objects in the image
2. Scene description and setting
3. Colors, lighting, and mood
4. Any text visible in the image
5. Technical aspects (composition, quality)
6. Any notable details or interesting elements

Be detailed but concise in your analysis.""",

            "basic": """Describe what you see in this image. Include:
- Main subject or focus
- Setting/background
- Key objects or elements
- Overall mood or atmosphere

Keep the description clear and concise.""",

            "text": """Focus on extracting and transcribing any text visible in this image. Include:
- All readable text (signs, labels, documents, etc.)
- Text location and context
- Language if not English
- Quality of text (clear, blurry, partial, etc.)

If no text is visible, clearly state that.""",

            "metadata": """Analyze the technical and contextual aspects of this image:
- Image quality and resolution apparent
- Lighting conditions
- Camera angle/perspective
- Possible location or setting type
- Time of day if determinable
- Any technical artifacts or issues

Focus on observable technical characteristics."""
        }

        prompt = prompts.get(analysis_type, prompts["comprehensive"])

        # Call Ollama API using chat endpoint (like PyQt app)
        ollama_url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False
        }

        with httpx.Client(timeout=300.0) as client:
            response = client.post(ollama_url, json=payload)

            if response.status_code != 200:
                return {
                    "tool": "image_analysis",
                    "success": False,
                    "error": f"Ollama API error: {response.status_code} - {response.text}"
                }

            result = response.json()
            analysis_text = result.get("message", {}).get("content", "No analysis provided")

        # Get basic file info
        file_stats = os.stat(image_path)
        file_size_mb = round(file_stats.st_size / (1024 * 1024), 2)

        # Determine image type for data URL
        file_extension = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(file_extension, 'image/jpeg')

        return {
            "tool": "image_analysis",
            "success": True,
            "image_path": image_path,
            "analysis_type": analysis_type,
            "model_used": model,
            "analysis": analysis_text,
            "file_info": {
                "filename": os.path.basename(image_path),
                "file_size_mb": file_size_mb,
                "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            },
            "image_data": {
                "base64": image_base64,
                "data_url": f"data:{mime_type};base64,{image_base64}",
                "mime_type": mime_type
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "tool": "image_analysis",
            "success": False,
            "error": f"Image analysis failed: {str(e)}"
        }

def analyze_image(image_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Main function for image analysis
    """
    return analyze_image_with_ollama(image_path, analysis_type)

def main():
    """CLI interface for the image analysis tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: image_analysis.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        image_path = args.get("image_path", "")
        analysis_type = args.get("analysis_type", "comprehensive")
        model = args.get("model", "gemma3:latest")

        if not image_path:
            print(json.dumps({"error": "image_path is required"}))
            sys.exit(1)

        result = analyze_image_with_ollama(image_path, analysis_type, model)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()