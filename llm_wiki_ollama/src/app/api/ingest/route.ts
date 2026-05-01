import { NextResponse } from 'next/server';
import PDFParser from 'pdf2json';
import fs from 'fs/promises';
import path from 'path';

const WIKI_DIR = path.join(process.cwd(), 'wiki');

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const model = formData.get('model') as string || 'gemma3';

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Ensure wiki directory exists
    await fs.mkdir(WIKI_DIR, { recursive: true });

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // 1. Extract Raw Text
    const rawText = await new Promise<string>((resolve, reject) => {
      const pdfParser = new (PDFParser as any)(null, 1);
      pdfParser.on("pdfParser_dataError", (errData: any) => reject(errData.parserError));
      pdfParser.on("pdfParser_dataReady", () => resolve(pdfParser.getRawTextContent()));
      pdfParser.parseBuffer(buffer);
    });

    // 2. Ingest: Ask LLM to convert to a structured Wiki Page
    const systemPrompt = `You are an expert knowledge curator and wiki maintainer. 
Your task is to read the provided raw document text and create a comprehensive, structured Markdown wiki page.
Do not output anything other than valid Markdown.
Include:
- A high-level Summary
- Key Concepts and Entities
- Important facts or decisions
Organize it cleanly using headings (##, ###) and bullet points.`;

    console.log("Generating wiki page using model:", model);
    const ollamaRes = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model,
        prompt: `Raw Document Text:\n${rawText.slice(0, 30000)}`, // Truncate if too large to avoid local OOM
        system: systemPrompt,
        stream: false,
      }),
    });

    if (!ollamaRes.ok) {
      throw new Error('Failed to communicate with local Ollama instance during ingestion.');
    }

    const data = await ollamaRes.json();
    const wikiMarkdown = data.response;

    // 3. Save to Wiki
    const fileName = file.name.replace(/\.[^/.]+$/, "") + ".md";
    const filePath = path.join(WIKI_DIR, fileName);
    await fs.writeFile(filePath, wikiMarkdown, 'utf-8');

    return NextResponse.json({ 
      message: 'Wiki page created successfully', 
      wikiText: wikiMarkdown,
      fileName 
    });

  } catch (error) {
    console.error('Error during ingestion:', error);
    return NextResponse.json(
      { error: 'Failed to ingest document and create wiki.' },
      { status: 500 }
    );
  }
}
