#!/usr/bin/env python3
"""
Image Analysis Tool
Analyzes image files for content, objects, text, and metadata
"""

import json
import sys
import os
from typing import Dict, Any, List
from datetime import datetime
import base64
import io

try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

def analyze_image(image_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Analyze image file for various properties and content

    Args:
        image_path: Path to image file
        analysis_type: Type of analysis (comprehensive, basic, text, metadata)

    Returns:
        Dictionary with analysis results
    """
    try:
        if not PIL_AVAILABLE:
            return {
                "tool": "image_analysis",
                "success": False,
                "error": "PIL (Pillow) library not available. Install with: pip install Pillow"
            }

        if not os.path.exists(image_path):
            return {
                "tool": "image_analysis",
                "success": False,
                "error": f"Image file not found: {image_path}"
            }

        # Open and analyze image
        with Image.open(image_path) as img:
            analysis_result = {
                "tool": "image_analysis",
                "success": True,
                "image_path": image_path,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }

            # Basic image properties
            basic_info = get_basic_image_info(img, image_path)
            analysis_result.update(basic_info)

            # Perform specific analysis based on type
            if analysis_type in ["comprehensive", "metadata"]:
                metadata = get_image_metadata(img)
                analysis_result["metadata"] = metadata

            if analysis_type in ["comprehensive", "text"] and TESSERACT_AVAILABLE:
                text_content = extract_text_from_image(img)
                analysis_result["text_content"] = text_content

            if analysis_type in ["comprehensive", "basic"]:
                visual_analysis = analyze_visual_content(img)
                analysis_result.update(visual_analysis)

            # Generate summary
            analysis_result["summary"] = generate_analysis_summary(analysis_result)

            return analysis_result

    except Exception as e:
        return {
            "tool": "image_analysis",
            "success": False,
            "error": str(e)
        }

def get_basic_image_info(img: Image.Image, image_path: str) -> Dict[str, Any]:
    """Extract basic image information"""
    file_size = os.path.getsize(image_path)

    return {
        "basic_info": {
            "filename": os.path.basename(image_path),
            "format": img.format,
            "mode": img.mode,
            "size": {
                "width": img.width,
                "height": img.height,
                "total_pixels": img.width * img.height
            },
            "file_size": {
                "bytes": file_size,
                "kb": round(file_size / 1024, 2),
                "mb": round(file_size / (1024 * 1024), 2)
            },
            "aspect_ratio": round(img.width / img.height, 2),
            "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
        }
    }

def get_image_metadata(img: Image.Image) -> Dict[str, Any]:
    """Extract EXIF and other metadata from image"""
    metadata = {
        "exif_data": {},
        "icc_profile": None,
        "other_info": {}
    }

    # Extract EXIF data
    if hasattr(img, '_getexif') and img._getexif() is not None:
        exif_data = img._getexif()
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            try:
                # Convert bytes to string if possible
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8')
                    except:
                        value = str(value)
                metadata["exif_data"][tag] = value
            except:
                metadata["exif_data"][tag] = str(value)

    # Extract other info
    if img.info:
        for key, value in img.info.items():
            if key not in ['exif']:
                try:
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    metadata["other_info"][key] = value
                except:
                    metadata["other_info"][key] = str(value)

    return metadata

def extract_text_from_image(img: Image.Image) -> Dict[str, Any]:
    """Extract text content using OCR"""
    if not TESSERACT_AVAILABLE:
        return {
            "available": False,
            "error": "pytesseract not available. Install with: pip install pytesseract"
        }

    try:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img

        # Extract text
        text = pytesseract.image_to_string(img_rgb)

        # Get detailed info
        text_data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)

        # Filter out low confidence text
        confident_text = []
        for i, confidence in enumerate(text_data['conf']):
            if int(confidence) > 30:  # Confidence threshold
                word = text_data['text'][i].strip()
                if word:
                    confident_text.append({
                        "text": word,
                        "confidence": int(confidence),
                        "position": {
                            "x": text_data['left'][i],
                            "y": text_data['top'][i],
                            "width": text_data['width'][i],
                            "height": text_data['height'][i]
                        }
                    })

        return {
            "available": True,
            "raw_text": text.strip(),
            "word_count": len(text.split()),
            "line_count": len([line for line in text.split('\n') if line.strip()]),
            "confident_words": confident_text,
            "has_text": bool(text.strip())
        }

    except Exception as e:
        return {
            "available": True,
            "error": f"OCR failed: {str(e)}"
        }

def analyze_visual_content(img: Image.Image) -> Dict[str, Any]:
    """Analyze visual characteristics of the image"""
    # Convert to RGB for consistent analysis
    if img.mode != 'RGB':
        img_rgb = img.convert('RGB')
    else:
        img_rgb = img

    # Analyze colors
    colors = img_rgb.getcolors(maxcolors=256*256*256)
    if colors:
        # Sort by frequency
        colors.sort(reverse=True)
        dominant_colors = []
        for count, color in colors[:5]:  # Top 5 colors
            dominant_colors.append({
                "rgb": color,
                "hex": "#{:02x}{:02x}{:02x}".format(*color),
                "percentage": round((count / (img.width * img.height)) * 100, 2)
            })
    else:
        dominant_colors = []

    # Calculate brightness
    brightness = calculate_brightness(img_rgb)

    # Detect if image is likely a photo, diagram, text, etc.
    image_type = classify_image_type(img_rgb)

    return {
        "visual_analysis": {
            "dominant_colors": dominant_colors,
            "brightness": {
                "average": brightness,
                "category": "bright" if brightness > 128 else "dark"
            },
            "estimated_type": image_type,
            "complexity": estimate_complexity(img_rgb)
        }
    }

def calculate_brightness(img: Image.Image) -> float:
    """Calculate average brightness of image"""
    # Convert to grayscale and calculate mean
    grayscale = img.convert('L')
    pixels = list(grayscale.getdata())
    return sum(pixels) / len(pixels)

def classify_image_type(img: Image.Image) -> str:
    """Estimate the type of image based on visual characteristics"""
    # This is a simplified classification
    # In practice, you'd use more sophisticated ML models

    brightness = calculate_brightness(img)

    # Simple heuristics
    if brightness > 200:
        return "document/text"
    elif brightness < 50:
        return "dark/artistic"
    else:
        return "photo/general"

def estimate_complexity(img: Image.Image) -> str:
    """Estimate visual complexity of the image"""
    # Simple edge detection approximation
    try:
        # Convert to grayscale
        gray = img.convert('L')

        # Resize for faster processing
        gray_small = gray.resize((100, 100))

        # Calculate variance (proxy for complexity)
        pixels = list(gray_small.getdata())
        mean = sum(pixels) / len(pixels)
        variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)

        if variance > 2000:
            return "high"
        elif variance > 500:
            return "medium"
        else:
            return "low"
    except:
        return "unknown"

def generate_analysis_summary(analysis_result: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the analysis"""
    basic = analysis_result.get("basic_info", {})
    visual = analysis_result.get("visual_analysis", {})
    text = analysis_result.get("text_content", {})

    summary_parts = []

    # Basic info
    if basic:
        size_info = basic.get("size", {})
        file_info = basic.get("file_size", {})
        summary_parts.append(
            f"Image: {basic.get('filename', 'Unknown')} "
            f"({size_info.get('width', 0)}x{size_info.get('height', 0)}, "
            f"{file_info.get('mb', 0)} MB, {basic.get('format', 'Unknown')} format)"
        )

    # Text content
    if text and text.get("available") and text.get("has_text"):
        word_count = text.get("word_count", 0)
        summary_parts.append(f"Contains {word_count} words of extractable text")
    elif text and text.get("available"):
        summary_parts.append("No readable text detected")

    # Visual characteristics
    if visual:
        brightness = visual.get("brightness", {})
        complexity = visual.get("complexity", "unknown")
        image_type = visual.get("estimated_type", "unknown")
        summary_parts.append(
            f"Visual characteristics: {brightness.get('category', 'unknown')} brightness, "
            f"{complexity} complexity, appears to be {image_type}"
        )

    return ". ".join(summary_parts) + "."

def main():
    """CLI interface for the image analysis tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: image_analysis.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        image_path = args.get("image_path", "")
        analysis_type = args.get("analysis_type", "comprehensive")

        if not image_path:
            print(json.dumps({"error": "image_path is required"}))
            sys.exit(1)

        result = analyze_image(image_path, analysis_type)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()