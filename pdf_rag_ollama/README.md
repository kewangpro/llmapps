# PDF RAG with Ollama

A Retrieval-Augmented Generation (RAG) application that allows users to chat with PDF documents and web content using local LLM models via Ollama. The application features document processing, vector storage, semantic search, and intelligent re-ranking for accurate question answering.

## Features

- **PDF Document Processing**: Upload and process PDF files for question answering
- **Web Content Processing**: Process web URLs for content-based queries
- **Vector Storage**: Uses ChromaDB for persistent vector storage with semantic embeddings
- **Intelligent Re-ranking**: Cross-encoder model for improved relevance ranking
- **Streamlit Interface**: User-friendly web interface for document upload and chat
- **Local LLM Integration**: Powered by Ollama for privacy-focused local inference

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT PROCESSING                                  │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │     UnstructuredPDFLoader       │  │     UnstructuredURLLoader       │   │
│  │          (PDF Files)            │  │        (Web Content)            │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
│                                    │                                        │
│                   RecursiveCharacterTextSplitter                           │
│                        (chunk_size=400, overlap=100)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR STORAGE (vector.py)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         ChromaDB                                   │   │
│  │              Persistent Storage: ./rag-chroma/                     │   │
│  │                                                                    │   │
│  │  ┌───────────────┐              ┌─────────────────────────────┐    │   │
│  │  │   Documents   │◄────────────►│  Ollama Embeddings          │    │   │
│  │  │   Metadata    │              │  (nomic-embed-text:latest)  │    │   │
│  │  │   IDs         │              │  via Ollama API             │    │   │
│  │  └───────────────┘              └─────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUERY PROCESSING                                    │
│                                                                             │
│  User Question ──► Vector Search (ChromaDB) ──► Retrieved Documents        │
│                         │                              │                   │
│                         ▼                              ▼                   │
│                Top 10 Results                  Re-ranking (ranker.py)      │
│                                                        │                   │
│                                        Cross-Encoder Model                 │
│                                     (ms-marco-MiniLM-L-6-v2)               │
│                                                        │                   │
│                                                        ▼                   │
│                                                 Top 3 Most Relevant        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLM GENERATION (llm.py)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Ollama Client                              │   │
│  │                    (http://localhost:11434)                        │   │
│  │                                                                    │   │
│  │  ┌───────────────────┐              ┌─────────────────────────┐    │   │
│  │  │   Context +       │────────────► │    LLM Model           │    │   │
│  │  │   User Question   │              │  (gemma3:latest)       │    │   │
│  │  │   System Prompt   │              │                        │    │   │
│  │  └───────────────────┘              └─────────────────────────┘    │   │
│  │                                                 │                   │   │
│  │                                                 ▼                   │   │
│  │                                        Streaming Response           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE DISPLAY                                   │
│                     Chat Interface with Retrieved Context                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

The application consists of four main components:

1. **`app.py`** - Main Streamlit application with UI and orchestration
2. **`vector.py`** - Vector database operations using ChromaDB
3. **`llm.py`** - Local LLM integration with Ollama
4. **`ranker.py`** - Document re-ranking using cross-encoder models

## Prerequisites

### Required Software
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally

### Required Ollama Models
```bash
# Install required models
ollama pull gemma3:latest
ollama pull nomic-embed-text:latest
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf_rag_ollama
```

2. Create and activate a virtual environment:
```bash
python -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start Ollama (if not already running):
```bash
ollama serve
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

### Using the Application

#### Upload PDF Documents
1. Use the sidebar to upload PDF files
2. Click "Upload" to process and add to vector store
3. Documents are chunked and embedded for semantic search

#### Process Web URLs
1. Enter a web URL in the sidebar
2. Click "Process" to extract and process content
3. Web content is added to the same vector store

#### Ask Questions
1. Type questions in the chat interface
2. The system retrieves relevant chunks from your documents
3. A cross-encoder re-ranks results for relevance
4. Local LLM generates answers based on retrieved context

## Configuration

### LLM Model
Edit `llm.py` to change the model:
```python
model="gemma3:latest"  # Change to your preferred model
```

### Embedding Model
Edit `vector.py` to change the embedding model:
```python
model_name="nomic-embed-text:latest"  # Change embedding model
```

### Chunking Parameters
Modify `app.py` for different text chunking:
```python
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=400,        # Adjust chunk size
    chunk_overlap=100,     # Adjust overlap
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)
```

## Technical Details

### Vector Storage
- **Database**: ChromaDB with persistent storage in `./rag-chroma/`
- **Embeddings**: Ollama's nomic-embed-text model
- **Distance Metric**: Cosine similarity

### Document Processing
- **PDF Loader**: UnstructuredPDFLoader for PDF parsing
- **URL Loader**: UnstructuredURLLoader for web content
- **Text Splitting**: Recursive character splitting with overlap

### Re-ranking
- **Model**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Top-K**: Returns top 3 most relevant chunks
- **Purpose**: Improves retrieval accuracy over pure vector similarity

### LLM Integration
- **Client**: Ollama Python client
- **Streaming**: Real-time response streaming
- **Context**: Structured prompts with retrieved context

## File Structure

```
pdf_rag_ollama/
├── app.py              # Main Streamlit application
├── llm.py              # LLM integration with Ollama
├── vector.py           # Vector database operations
├── ranker.py           # Cross-encoder re-ranking
├── requirements.txt    # Python dependencies
├── myvenv/            # Virtual environment
└── rag-chroma/        # ChromaDB persistent storage
    ├── chroma.sqlite3
    └── <embeddings>/
```

## Dependencies

- **ollama**: Ollama Python client
- **chromadb**: Vector database
- **sentence-transformers**: Cross-encoder models
- **streamlit**: Web interface
- **langchain-community**: Document loaders
- **unstructured[local-inference]**: PDF/URL processing

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`

2. **Missing Models**
   ```bash
   ollama pull gemma3:latest
   ollama pull nomic-embed-text:latest
   ```

3. **Memory Issues**
   - Reduce chunk_size in text splitter
   - Use smaller LLM models
   - Increase system RAM allocation

4. **Slow Performance**
   - Reduce n_results in vector queries
   - Lower top_k in re-ranking
   - Use faster embedding models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.