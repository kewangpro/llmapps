# Ollama LLM-Wiki

Ollama LLM-Wiki is a modern, lightweight Next.js web application designed to help you interact with your PDF documents locally. It leverages the power of local LLMs running via [Ollama](https://ollama.com) to provide a private, completely offline "chat with your document" experience.

Inspired by the [LLM-Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), this application extracts knowledge from your PDFs and uses it as context to accurately synthesize answers using your preferred local AI models.

## ✨ Features

- **Local & Private**: All data processing and LLM querying happens locally on your machine.
- **PDF Text Extraction**: Uses `pdf2json` to reliably extract text content from your uploaded PDF documents.
- **Dynamic Model Selection**: Automatically fetches your local Ollama models and defaults to `gemma3`. You can choose any model you have installed.
- **Modern UI/UX**: A beautiful, light-themed three-column interface built with Tailwind CSS, featuring subtle borders, clean typography, and responsive micro-animations.

## 🛠️ Tech Stack

- **Framework**: [Next.js 15](https://nextjs.org/) (App Router)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **PDF Parsing**: [pdf2json](https://www.npmjs.com/package/pdf2json)
- **AI Backend**: [Ollama](https://ollama.com) (Local LLM API)

## 🚀 Getting Started

### Prerequisites

1. **Node.js** (v18 or higher)
2. **Ollama**: Make sure you have Ollama installed and running locally on port `11434`. You should also have at least one model pulled (e.g., `ollama run llama3`).

### Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   cd llm_wiki_ollama
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to [http://localhost:3000](http://localhost:3000).

## 💡 How to Use

1. **Select a Model**: Use the dropdown in the top right corner to select your preferred local Ollama model (defaults to `gemma3` if available).
2. **Manage Knowledge Base**: Click the upload area on the left to select a PDF file. The app will extract the text, synthesize it, and add it to your "Processed Documents". You can hover over any document in the list to delete it from the knowledge base.
3. **Explore the Wiki**: The right sidebar provides a live, truncated preview of each generated Markdown document in your current knowledge base.
4. **Ask Questions**: Once documents are loaded, use the central chat interface to ask questions. The local LLM will synthesize an answer using the compiled wiki text as its grounding context.

## 📝 License

This project is licensed under the MIT License.
