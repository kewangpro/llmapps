import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const WIKI_DIR = path.join(process.cwd(), 'wiki');

export async function GET() {
  try {
    const files = await fs.readdir(WIKI_DIR);
    let combinedWiki = "";
    let documents = [];
    
    for (const file of files) {
      if (file.endsWith('.md')) {
        const content = await fs.readFile(path.join(WIKI_DIR, file), 'utf-8');
        combinedWiki += `## ${file}\n\n${content}\n\n`;
        documents.push({
          title: file.replace('.md', ''),
          content: content
        });
      }
    }
    
    return NextResponse.json({ 
      text: combinedWiki, 
      files: files.filter(f => f.endsWith('.md')),
      documents: documents
    });
  } catch (error) {
    // Return empty if directory doesn't exist yet
    return NextResponse.json({ text: "", files: [], documents: [] });
  }
}

export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const file = searchParams.get('file');

    if (!file || !file.endsWith('.md')) {
      return NextResponse.json({ error: 'Invalid file parameter' }, { status: 400 });
    }

    const filePath = path.join(WIKI_DIR, file);
    await fs.unlink(filePath);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to delete file' }, { status: 500 });
  }
}
