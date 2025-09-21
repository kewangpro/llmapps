#!/usr/bin/env python3
"""
Image Analysis Tool
Processes and reads image files, extracting metadata and content
"""

import json
import sys
import os
from typing import Dict, Any
from datetime import datetime
import base64
import mimetypes

def encode_image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """Convert image file to base64 encoding with size optimization"""
    try:
        # First check file size
        file_size = os.path.getsize(image_path)

        # If file is very large (>2MB), resize it
        if file_size > 2 * 1024 * 1024:  # 2MB
            try:
                from PIL import Image
                import io

                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize to max dimensions while maintaining aspect ratio
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                    # Save to bytes with compression
                    output = io.BytesIO()
                    img.save(output, format='JPEG', quality=75, optimize=True)
                    return base64.b64encode(output.getvalue()).decode('utf-8')

            except ImportError:
                # PIL not available, fall back to original file
                pass

        # Use original file if small enough or PIL not available
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        raise Exception(f"Failed to encode image: {str(e)}")

def process_image_file(image_path: str) -> Dict[str, Any]:
    """
    Process image file and extract metadata, content, and base64 data
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                "tool": "image_analysis",
                "success": False,
                "error": f"Image file not found: {image_path}"
            }

        # Get file info
        file_stats = os.stat(image_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default fallback

        # Encode image to base64
        try:
            image_base64 = encode_image_to_base64(image_path)
        except Exception as e:
            return {
                "tool": "image_analysis",
                "success": False,
                "error": f"Failed to process image: {str(e)}"
            }

        # Extract additional metadata if PIL is available
        additional_metadata = {}
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            with Image.open(image_path) as img:
                additional_metadata = {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format
                }

                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif_data = {}
                    for tag_id, value in img._getexif().items():
                        tag = TAGS.get(tag_id, tag_id)
                        # Convert non-JSON serializable types to strings
                        try:
                            import json
                            json.dumps(value)  # Test if value is JSON serializable
                            exif_data[tag] = value
                        except (TypeError, ValueError):
                            # Convert to string if not serializable
                            exif_data[tag] = str(value)
                    additional_metadata["exif"] = exif_data

        except ImportError:
            # PIL not available, continue without additional metadata
            pass
        except Exception:
            # Error reading image metadata, continue without it
            pass

        return {
            "tool": "image_analysis",
            "success": True,
            "image_path": image_path,
            "file_info": {
                "filename": os.path.basename(image_path),
                "file_size_mb": file_size_mb,
                "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                **additional_metadata
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

def main():
    """CLI interface for the image analysis tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: image_analysis.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        image_path = args.get("image_path", "")

        if not image_path:
            print(json.dumps({"error": "image_path is required"}))
            sys.exit(1)

        result = process_image_file(image_path)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()