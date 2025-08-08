import sys
import asyncio
import PyPDF2
import aiohttp
import re
import os
from pptx import Presentation

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

async def process_pdf(pdf_path, progress_cb=None):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        msg = f'❌ Failed to open PDF: {e}'
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)
        return

    # Print the extracted text for debugging (first 500 chars)
    debug_text = text[:500] + ("..." if len(text) > 500 else "")
    msg = f'--- Extracted PDF text (first 500 chars) ---\n{debug_text}\n--- End of extract ---'
    if progress_cb:
        progress_cb(msg)
    else:
        print(msg)

    if not text.strip():
        msg = '⚠️ Warning: Extracted PDF text is empty.'
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)
        return

    # Chunking utility
    def chunk_text(text, max_length=3000):
        # Split text into chunks of max_length characters, on paragraph boundaries if possible
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

    # If text is too long, chunk it and process each chunk separately
    max_chunk_length = 3000  # adjust as needed for your LLM
    text_chunks = chunk_text(text, max_chunk_length)
    all_slides = []
    for chunk_idx, chunk in enumerate(text_chunks):
        chunk_prompt = f'''
You are a presentation assistant. Your ONLY task is to convert the following document into a slide outline.

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

Document:
{chunk}
'''
        msg = f'⏳ Sending request to LLM for chunk {chunk_idx+1}/{len(text_chunks)}...'
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post('http://localhost:11434/api/generate', json={
                    'model': 'gemma3:latest',
                    'prompt': chunk_prompt,
                    'stream': False
                }) as resp:
                    result = await resp.json()
        except Exception as e:
            msg = f'❌ LLM API error: {e}'
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)
            return

        if 'response' not in result:
            msg = f'⚠️ Unexpected API response: {result}'
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)
            return

        outline = result['response']
        msg = f'=== LLM Output Start (chunk {chunk_idx+1}) ===\n' + outline + f'\n=== LLM Output End (chunk {chunk_idx+1}) ==='
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

        slides_raw = re.split(r'Slide\s*\d+:', outline, flags=re.IGNORECASE)[1:]
        if not slides_raw or all(not s.strip() for s in slides_raw):
            msg = (f"❌ LLM output for chunk {chunk_idx+1} did not contain any slides in the expected format. "
                   "Please check your LLM prompt and model output.\n"
                   "First 500 chars of LLM output:\n" + outline[:500] + ("..." if len(outline) > 500 else ""))
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)
            continue  # skip this chunk
        all_slides.extend(slides_raw)

    if not all_slides:
        msg = ("❌ No valid slides were generated from any chunk. "
               "Please check your LLM prompt, model, and PDF content.")
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)
        return

    prs = Presentation()
    for idx, slide_text in enumerate(all_slides, 1):
        lines = [line.strip() for line in slide_text.strip().splitlines() if line.strip()]
        title = None
        bullets = []
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
        msg = f'✅ Added Slide {idx}: "{title}" with {len(bullets)} bullet points.'
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = f'{pdf_name}_presentation.pptx'
    prs.save(output_file)
    msg = f'🎉 Presentation saved as {output_file}'
    if progress_cb:
        progress_cb(msg)
    else:
        print(msg)
    return
    # ...existing code...
class Worker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, pdf_path):
        super().__init__()
        self.pdf_path = pdf_path

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        await process_pdf(self.pdf_path, self.progress.emit)
        self.finished.emit("Done")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF to PPTX with Ollama LLM")
        self.setGeometry(300, 300, 600, 400)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Select a PDF file to generate a presentation.")
        self.layout.addWidget(self.label)

        self.select_btn = QPushButton("Select PDF")
        self.select_btn.clicked.connect(self.select_pdf)
        self.layout.addWidget(self.select_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)

        self.worker = None

    def select_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.text_edit.clear()
            self.progress_bar.setVisible(True)
            self.label.setText(f"Processing: {file_path}")
            self.worker = Worker(file_path)
            self.worker.progress.connect(self.append_text)
            self.worker.finished.connect(self.done)
            self.worker.start()

    def append_text(self, msg):
        self.text_edit.append(msg)

    def done(self, msg):
        self.progress_bar.setVisible(False)
        self.text_edit.append(msg)
        self.label.setText("Done! Select another PDF to convert.")

def main():
    # CLI mode
    if '--pdf_file' in sys.argv:
        idx = sys.argv.index('--pdf_file')
        if idx + 1 < len(sys.argv):
            pdf_path = sys.argv[idx + 1]
            asyncio.run(process_pdf(pdf_path))
        else:
            print("Usage: python gen_ppt.py --pdf_file <path_to_pdf>")
        return

    # GUI mode
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
