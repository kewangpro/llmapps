#!/usr/bin/env python3
"""
Presentation Tool
Generates PowerPoint presentations from input text or files using Ollama LLM
"""

import json
import sys
import os
import asyncio
import aiohttp
import re
import base64
from typing import Dict, Any, List
from datetime import datetime

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

async def generate_presentation_with_ollama(input_text: str, title: str = "", output_filename: str = "presentation.pptx", model: str = "gemma3:latest") -> Dict[str, Any]:
    """
    Generate PowerPoint presentation from input text using Ollama LLM

    Args:
        input_text: Text content to convert to presentation
        title: Presentation title
        output_filename: Output file name
        model: Ollama model to use

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

        # Chunk text if it's too long
        max_chunk_length = 3000
        text_chunks = chunk_text(input_text, max_chunk_length)
        all_slides = []

        for chunk_idx, chunk in enumerate(text_chunks):
            chunk_prompt = f'''
You are a presentation assistant. Your ONLY task is to create a slide outline based on the following content.

IMPORTANT: DO NOT provide any commentary, review, explanation, or text before or after the slides. If you do, the user's program will break.

INSTRUCTIONS:
- Output ONLY slides in the following format. Do NOT add any extra text, commentary, or explanation.
- Always generate at least 3 slides.
- Each slide must have a title and 3 to 5 bullet points.
- Use this exact format (do not change it):

Slide 1:
Title: <title of slide 1>
- <bullet point 1>
- <bullet point 2>
- <bullet point 3>
Slide 2:
Title: <title of slide 2>
- <bullet point 1>
- <bullet point 2>
- <bullet point 3>
Slide 3:
Title: <title of slide 3>
- <bullet point 1>
- <bullet point 2>
- <bullet point 3>

Continue for as many slides as needed, but do not add any text outside this format. Do NOT include any summary, review, or extra lines.

Content:
{chunk}
'''

            # Call Ollama API
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post('http://localhost:11434/api/generate', json={
                        'model': model,
                        'prompt': chunk_prompt,
                        'stream': False
                    }) as resp:
                        result = await resp.json()
            except Exception as e:
                return {
                    "tool": "presentation",
                    "success": False,
                    "error": f"LLM API error: {e}"
                }

            if 'response' not in result:
                return {
                    "tool": "presentation",
                    "success": False,
                    "error": f"Unexpected API response: {result}"
                }

            outline = result['response']

            # Parse slides from LLM output
            slides_raw = re.split(r'Slide\s*\d+:', outline, flags=re.IGNORECASE)[1:]
            if not slides_raw or all(not s.strip() for s in slides_raw):
                # Continue with next chunk if this one fails
                continue
            all_slides.extend(slides_raw)

        if not all_slides:
            return {
                "tool": "presentation",
                "success": False,
                "error": "No valid slides were generated from input text"
            }

        # Create PowerPoint presentation
        prs = Presentation()
        prs.slide_width = Inches(13.33)
        prs.slide_height = Inches(7.5)

        slides_created = 0
        slide_descriptions = []

        # Add title slide if title provided
        if title:
            title_slide = prs.slides.add_slide(prs.slide_layouts[0])
            title_slide.shapes.title.text = title
            slides_created += 1
            slide_descriptions.append({
                "title": title,
                "bullets": [],
                "slide_type": "title"
            })

        # Process each slide
        for idx, slide_text in enumerate(all_slides, 1):
            lines = [line.strip() for line in slide_text.strip().splitlines() if line.strip()]
            slide_title = None
            bullets = []

            # Parse title and bullets
            for i, line in enumerate(lines):
                if line.lower().startswith('title:'):
                    slide_title = line[len('title:'):].strip()
                    bullets = [l[1:].strip() for l in lines[i+1:] if l.startswith('-')]
                    break

            if not slide_title:
                if lines:
                    slide_title = lines[0]
                    bullets = [l[1:].strip() for l in lines[1:] if l.startswith('-')]
                else:
                    slide_title = f"Slide {idx}"
                    bullets = []

            # Create slide
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = slide_title

            # Add content
            if len(slide.placeholders) > 1 and bullets:
                slide.placeholders[1].text = '\n'.join(bullets)

            # Store slide description
            slide_descriptions.append({
                "title": slide_title,
                "bullets": bullets,
                "slide_type": "content"
            })

            slides_created += 1

        # Save presentation
        output_path = os.path.join("outputs", output_filename)
        os.makedirs("outputs", exist_ok=True)
        prs.save(output_path)

        # Encode PPT file to base64 for frontend display
        with open(output_path, "rb") as ppt_file:
            ppt_base64 = base64.b64encode(ppt_file.read()).decode('utf-8')

        return {
            "tool": "presentation",
            "success": True,
            "output_file": output_path,
            "slides_created": slides_created,
            "total_slides": len(all_slides) + (1 if title else 0),
            "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
            "message": f"Generated presentation with {slides_created} slides: {output_path}",
            "presentation_data": {
                "base64": ppt_base64,
                "filename": output_filename,
                "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "slides": slide_descriptions
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "tool": "presentation",
            "success": False,
            "error": str(e)
        }

def chunk_text(text: str, max_length: int = 3000) -> List[str]:
    """Splits text into chunks of max_length characters, trying to preserve paragraphs."""
    paragraphs = text.split('\n')
    chunks = []
    current = ''
    for para in paragraphs:
        if len(current) + len(para) + 1 > max_length:
            if current:
                chunks.append(current)
            current = para
        else:
            current += ('\n' if current else '') + para
    if current:
        chunks.append(current)
    return chunks

def generate_presentation(input_text: str, title: str = "", output_filename: str = "presentation.pptx") -> Dict[str, Any]:
    """
    Synchronous wrapper for generate_presentation_with_ollama
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            generate_presentation_with_ollama(input_text, title, output_filename)
        )
        loop.close()
        return result
    except Exception as e:
        return {
            "tool": "presentation",
            "success": False,
            "error": str(e)
        }


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