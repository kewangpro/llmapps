#!/usr/bin/env python3
"""
Presentation Tool
Generates PowerPoint presentations from input text or files
"""

import json
import sys
import os
from typing import Dict, Any, List
from datetime import datetime
import re

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

def generate_presentation(input_text: str, title: str = "", output_filename: str = "presentation.pptx") -> Dict[str, Any]:
    """
    Generate PowerPoint presentation from input text

    Args:
        input_text: Text content to convert to presentation
        title: Presentation title
        output_filename: Output file name

    Returns:
        Dictionary with generation results
    """
    try:
        if not PPTX_AVAILABLE:
            return {
                "tool": "presentation",
                "success": False,
                "error": "python-pptx library not available. Install with: pip install python-pptx"
            }

        # Parse input text to extract slides
        slides_data = parse_text_to_slides(input_text, title)

        # Create presentation
        prs = Presentation()

        # Set slide size (16:9 widescreen)
        prs.slide_width = Inches(13.33)
        prs.slide_height = Inches(7.5)

        slides_created = 0

        for i, slide_data in enumerate(slides_data):
            slide_layout = prs.slide_layouts[1]  # Title and Content layout
            if i == 0:
                slide_layout = prs.slide_layouts[0]  # Title slide layout

            slide = prs.slides.add_slide(slide_layout)

            # Add title
            if slide.shapes.title:
                slide.shapes.title.text = slide_data["title"]

                # Style the title
                title_paragraph = slide.shapes.title.text_frame.paragraphs[0]
                title_paragraph.font.size = Pt(32) if i == 0 else Pt(28)
                title_paragraph.font.bold = True
                title_paragraph.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue

            # Add content
            if len(slide.placeholders) > 1 and slide_data["content"]:
                content_placeholder = slide.placeholders[1]
                text_frame = content_placeholder.text_frame
                text_frame.clear()

                # Add bullet points
                for j, point in enumerate(slide_data["content"]):
                    p = text_frame.paragraphs[0] if j == 0 else text_frame.add_paragraph()
                    p.text = point
                    p.level = 0
                    p.font.size = Pt(18)
                    p.space_before = Pt(6)

            slides_created += 1

        # Save presentation
        output_path = os.path.join("outputs", output_filename)
        os.makedirs("outputs", exist_ok=True)
        prs.save(output_path)

        return {
            "tool": "presentation",
            "success": True,
            "output_file": output_path,
            "slides_created": slides_created,
            "total_slides": len(slides_data),
            "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
            "message": f"Generated presentation with {slides_created} slides: {output_path}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "tool": "presentation",
            "success": False,
            "error": str(e)
        }

def parse_text_to_slides(text: str, presentation_title: str = "") -> List[Dict[str, Any]]:
    """Parse text content into slide structure"""
    slides = []

    # Split text into sections
    lines = text.split('\n')
    current_slide = {
        "title": presentation_title or "Presentation",
        "content": []
    }

    # Add title slide if presentation title is provided
    if presentation_title:
        slides.append({
            "title": presentation_title,
            "content": []
        })

    slide_title_patterns = [
        r'^#+\s+(.+)',  # Markdown headers
        r'^([A-Z][A-Za-z\s]{3,}):?\s*$',  # Title case lines
        r'^\d+\.\s+(.+)',  # Numbered sections
        r'^[•\-\*]\s*([A-Z][A-Za-z\s]{5,})',  # Bullet points that look like titles
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this line looks like a slide title
        is_title = False
        for pattern in slide_title_patterns:
            match = re.match(pattern, line)
            if match:
                # Save current slide if it has content
                if current_slide["content"] or (current_slide["title"] and current_slide["title"] != presentation_title):
                    slides.append(current_slide)

                # Start new slide
                current_slide = {
                    "title": match.group(1) if match.lastindex else line.replace('#', '').strip(),
                    "content": []
                }
                is_title = True
                break

        if not is_title:
            # Add as content to current slide
            # Clean up bullet points
            cleaned_line = re.sub(r'^[•\-\*\+]\s*', '', line)
            if cleaned_line:
                current_slide["content"].append(cleaned_line)

    # Add the last slide
    if current_slide["content"] or current_slide["title"]:
        slides.append(current_slide)

    # If no clear structure found, create slides based on paragraphs
    if len(slides) <= 1 and not presentation_title:
        slides = create_slides_from_paragraphs(text)

    # Limit bullet points per slide
    final_slides = []
    for slide in slides:
        if len(slide["content"]) > 8:  # Too many points for one slide
            # Split into multiple slides
            chunks = [slide["content"][i:i+6] for i in range(0, len(slide["content"]), 6)]
            for i, chunk in enumerate(chunks):
                title_suffix = f" (Part {i+1})" if len(chunks) > 1 else ""
                final_slides.append({
                    "title": slide["title"] + title_suffix,
                    "content": chunk
                })
        else:
            final_slides.append(slide)

    return final_slides

def create_slides_from_paragraphs(text: str) -> List[Dict[str, Any]]:
    """Create slides from paragraph breaks when no clear structure exists"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    slides = []

    for i, paragraph in enumerate(paragraphs):
        # Use first sentence as title, rest as content
        sentences = paragraph.split('.')
        title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
        content = ['. '.join(sentences[1:]).strip()] if len(sentences) > 1 else []

        slides.append({
            "title": title or f"Slide {i+1}",
            "content": content
        })

    return slides

def main():
    """CLI interface for the presentation tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: presentation.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        input_text = args.get("input_text", "")
        title = args.get("title", "Generated Presentation")
        output_filename = args.get("output_filename", "presentation.pptx")

        if not input_text:
            print(json.dumps({"error": "input_text is required"}))
            sys.exit(1)

        result = generate_presentation(input_text, title, output_filename)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()