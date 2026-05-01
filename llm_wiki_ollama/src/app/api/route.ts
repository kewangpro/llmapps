import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const response = await fetch('http://localhost:11434/api/tags');
    
    if (!response.ok) {
      throw new Error('Failed to fetch models');
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching models:', error);
    return NextResponse.json(
      { error: 'Could not connect to local Ollama instance.' },
      { status: 500 }
    );
  }
}
