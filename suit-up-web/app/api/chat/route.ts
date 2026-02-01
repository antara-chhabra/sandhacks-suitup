import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const response = await fetch('http://127.0.0.1:11434/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3',
        messages,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama responded with ${response.status}`);
    }
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Ollama Error:', error);
    return NextResponse.json(
      { error: 'Ollama connection failed. Is it running?' },
      { status: 500 }
    );
  }
}
