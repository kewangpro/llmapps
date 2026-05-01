# System Design

This document explains the architecture, design decisions, and data flow of the Ollama LLM-Wiki application.

## 🏗 Architecture Overview

The application follows a simple, three-tier architecture using Next.js for both the presentation layer and the API layer, while delegating the heavy AI computation to a local Ollama instance. The backend interacts directly with the local file system to store compiled knowledge.

1. **Frontend (Client)**: A React-based Single Page Application (SPA) built with Next.js App Router and Tailwind CSS.
2. **Backend API (Server)**: Next.js API Routes acting as a bridge for file processing, wiki storage, and Ollama proxying.
3. **AI Backend**: An external, locally-hosted Ollama API service (`http://localhost:11434`).

```text
  +-----------------------+           +-----------------------+           +-----------------------+
  |    Browser (Client)   |           |    Next.js (Server)   |           |     Ollama (Local)    |
  |                       |           |                       |           |                       |
  |  +-----------------+  |           |  +-----------------+  |           |  +-----------------+  |
  |  |                 |  |  Upload   |  |                 |  |           |  |                 |  |
  |  |                 |  +----------->  |  POST /api/ingest |  +----------->  |                 |  |
  |  |                 |  |   PDF     |  |                 |  | Raw Text  |  |                 |  |
  |  |   React State   |  <-----------+  |  (1. pdf2json)  |  <-----------+  |   LLM Engine    |  |
  |  |                 |  |  Success  |  |  (2. FS Write)  |  | Wiki MD   |  | (gemma3, llama3)|  |
  |  |                 |  |           |  +-------+---------+  |           |  |                 |  |
  |  |                 |  |  Delete   |  +-------+---------+  |           |  |                 |  |
  |  |                 |  +----------->  |                 |  |           |  |                 |  |
  |  |                 |  |  Filename |  | DELETE /api/wiki|  |           |  |                 |  |
  |  +-----------------+  |  Success  |  |  (Unlink file)  |  |           |  |                 |  |
  |                       | <-----------+  +-------+---------+  |           |  |                 |  |
  |  +-----------------+  |           |          |            |           |  |                 |  |
  |  |                 |  |   Prompt  |  +-------v---------+  |  Context  |  |                 |  |
  |  |  Chat Interface |  +----------->  |                 |  +----------->  |                 |  |
  |  |                 |  <-----------+  |  POST /api/chat |  |           |  |                 |  |
  |  +-----------------+  |  Response |  | (Reads ./wiki)  |  <-----------+  |                 |  |
  |                       |           |  +-----------------+  |  Response |  +-----------------+  |
  +-----------------------+           +-----------------------+           +-----------------------+
```

## 🔄 Data Flow

The application flow can be broken down into three main phases: Setup, Ingestion, and Inference.

### 1. Setup (Model Discovery)
When the application loads, it needs to know what capabilities the local AI has.
- **Client** makes a `GET` request to `/api/models`.
- **API Route** proxies the request to `http://localhost:11434/api/tags`.
- **API Route** returns a list of available models to the Client, which populates the model selector dropdown.

### 2. Ingestion (Wiki Compilation)
Instead of relying on chunk-first vector databases (as discussed in the *LLM-Wiki pattern*), this app takes a synthesis-first approach.
- **Client** uploads a PDF file using the `FormData` API and sends a `POST` request to `/api/ingest`.
- **API Route** receives the binary buffer and extracts the raw text using `pdf2json` (a pure Node.js parser compatible with Next.js Turbopack).
- **API Route** sends the raw text to Ollama with a "Knowledge Curator" system prompt, instructing it to synthesize the text into a structured, comprehensive Markdown wiki page.
- The compiled Markdown is written to the local file system in a dedicated `./wiki` directory.
- The Client refreshes its state by fetching the compiled wiki via `GET /api/wiki`.
- The Client can also manage the wiki by sending a `DELETE /api/wiki?file=filename.md` request to remove documents from the disk and update the context.

### 3. Inference (Wiki-Grounded Chat)
When the user asks a question, the application uses the compiled wiki (not the raw text) as the grounding context.
- **Client** sends a `POST` request to `/api/chat` containing the `prompt` and the selected `model`.
- **API Route** reads all Markdown files present in the `./wiki` directory.
- **API Route** constructs a system prompt dynamically, passing the concatenated wiki content to ground the LLM's answers.
- **API Route** makes a `POST` request to `http://localhost:11434/api/generate`, initiating the local inference.
- The synthesized response is returned back through the Next.js API to the Client, where it is appended to the chat interface.

## 📚 Inspiration: The LLM-Wiki Pattern

This application's approach to knowledge management is heavily inspired by Andrej Karpathy's [LLM-Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). 

### The Traditional RAG Problem
Standard Retrieval-Augmented Generation (RAG) usually relies on a "chunk-first" architecture: documents are split into arbitrary chunks, embedded into a vector database, and fetched via semantic search at query time. This often leads to fragmented context and the LLM struggling to synthesize concepts that span multiple sections of a document.

### The LLM-Wiki Paradigm
The LLM-Wiki pattern proposes a different approach: instead of fragmenting raw sources, the LLM processes them to build and maintain a persistent, structured, and interlinked "wiki" of markdown files. The knowledge is compiled once, updated incrementally, and queried cohesively.

### How this App Adapts the Pattern
This project fully adopts the paradigm by moving away from traditional in-memory RAG toward a persistent disk-based knowledge graph:
- **Avoiding Chunks**: We extract the PDF and pass the entire text to the LLM to synthesize a new page, relying on large context windows rather than semantic search across fragments.
- **Persistent Artifacts**: The generated insights are not discarded. They are written as persistent Markdown files to a `./wiki` directory, creating a compounding artifact.
- **Wiki-Grounded Queries**: Future user queries are evaluated against the compiled, structured wiki, not the raw source documents, ensuring a much higher quality of synthesis.

## 🧠 Design Decisions & Trade-offs

### 1. Persistent Local Wiki
**Decision**: The application generates and writes Markdown files to the local disk instead of returning temporary context to the React frontend.
**Why**: 
- **Knowledge Compounding**: By writing to the `./wiki` folder, the knowledge base persists between sessions. Every new PDF uploaded adds a new page to the wiki.
- **Speed**: Querying pre-synthesized markdown is faster and yields higher-quality answers than sending raw PDF dumps to the LLM on every query.

### 2. Pure Node.js PDF Parsing (`pdf2json`)
**Decision**: Using `pdf2json` over `pdf-parse` or `pdfjs-dist`.
**Why**: Modern Next.js (especially when using Turbopack) enforces strict environments. Standard `pdf.js` implementations often expect browser globals (`DOMMatrix`, `canvas`) which break during the Next.js API route compilation. `pdf2json` circumvents this by strictly parsing the binary format in Node.js, ensuring stable production builds.

### 3. Full-Context Generation vs. Chunked Generation
**Decision**: The application passes the *entire* extracted text into the LLM's context window to generate the wiki page.
**Why**: 
- **Simplicity**: Eliminates the need for a vector database (like ChromaDB or Pinecone) and an embedding model.
- **Context Windows**: Modern local models (like `llama3`) have massive context windows, which can easily fit several pages of dense PDF text in a single prompt to generate a highly accurate summary.

## 🎨 UI/UX Philosophy

The interface was designed to feel premium and engaging, using the following principles:
- **Spatial Separation (Three Columns)**: The screen is cleanly split into three active zones: "Knowledge Ingestion & Curation" (Left), "Exploration/Chat" (Center), and "Wiki Preview" (Right).
- **Visual Feedback**: Granular states (`isUploading`, `uploadSuccess`, `isQuerying`) are tied to distinct micro-animations and color changes to keep the user informed.
- **Clean Light Theme**: Utilizing clean white backgrounds with subtle zinc borders and drop shadows (`bg-white`, `border-zinc-200`) against a light slate background creates a modern, highly readable aesthetic.
