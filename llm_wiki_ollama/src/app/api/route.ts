import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const WIKI_DIR = path.join(process.cwd(), 'wiki');

export async function POST(request: Request) {
  try {
    const { prompt, model } = await request.json();

    if (!prompt) {
      return NextResponse.json({ error: 'No prompt provided' }, { status: 400 });
    }

    // Load compiled wiki as context
    let wikiContext = "";
    try {
      const files = await fs.readdir(WIKI_DIR);
      for (const file of files) {
        if (file.endsWith('.md')) {
          const content = await fs.readFile(path.join(WIKI_DIR, file), 'utf-8');
          wikiContext += `\n--- Source: ${file} ---\n${content}\n`;
        }
      }
    } catch (err) {
      console.log('No wiki directory found or empty, querying without context.');
    }

    const systemPrompt = wikiContext 
      ? `You are a helpful assistant with access to a curated knowledge wiki. Use the following wiki pages to answer the user's question accurately. Do not invent facts outside of the wiki.\n\nWiki Context:\n${wikiContext}`
      : 'You are a helpful assistant.';

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model || 'gemma3',
        prompt: prompt,
        system: systemPrompt,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to communicate with local Ollama instance.');
    }

    const data = await response.json();
    return NextResponse.json({ response: data.response });

  } catch (error) {
    console.error('Error querying Ollama:', error);
    return NextResponse.json(
      { error: 'Failed to query Ollama.' },
      { status: 500 }
    );
  }
}
