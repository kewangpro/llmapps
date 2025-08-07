import asyncio
import PyPDF2
import aiohttp
import re
import sys
import argparse
import os
from pptx import Presentation

async def main(pdf_path):
    try:
        reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
    except Exception as e:
        print(f'❌ Failed to open PDF: {e}')
        return

    text = ''.join(page.extract_text() or '' for page in reader.pages)
    if not text.strip():
        print('⚠️ Warning: Extracted PDF text is empty.')
        return

    prompt = f'''
You are a presentation assistant. Summarize the following document into slides.
Each slide must have a title and 3 to 5 bullet points.
Output ONLY in the following exact format, no extra text:

Slide 1:
Title: ...
- ...
- ...
Slide 2:
Title: ...
- ...
- ...

Document:
{text}
'''

    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:11434/api/generate', json={
            'model': 'gemma3:latest',
            'prompt': prompt,
            'stream': False
        }) as resp:
            try:
                result = await resp.json()
            except Exception as e:
                print(f'❌ Failed to parse JSON response: {e}')
                return

    if 'response' not in result:
        print('⚠️ Unexpected API response:')
        print(result)
        return

    outline = result['response']
    print('\n=== LLM Output Start ===')
    print(outline)
    print('=== LLM Output End ===\n')

    slides_raw = re.split(r'Slide\s*\d+:', outline, flags=re.IGNORECASE)[1:]

    prs = Presentation()

    for idx, slide_text in enumerate(slides_raw, 1):
        lines = [line.strip() for line in slide_text.strip().splitlines() if line.strip()]
        title = None
        bullets = []

        # Find Title line
        for i, line in enumerate(lines):
            if line.lower().startswith('title:'):
                title = line[len('title:'):].strip()
                bullets = [l[1:].strip() for l in lines[i+1:] if l.startswith('-')]
                break

        if not title:
            if lines:
                title = lines[0]
                bullets = [l[1:].strip() for l in lines[1:] if l.startswith('-')]
            else:
                title = f"Slide {idx}"
                bullets = []

        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = '\n'.join(bullets)
        print(f'✅ Added Slide {idx}: "{title}" with {len(bullets)} bullet points.')

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = f'{pdf_name}_presentation.pptx'
    prs.save(output_file)
    print(f'🎉 Presentation saved as {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate presentation PPTX from PDF using Ollama LLM')
    parser.add_argument('--pdf_file', required=True, help='Path to the PDF file to process')
    args = parser.parse_args()

    asyncio.run(main(args.pdf_file))
