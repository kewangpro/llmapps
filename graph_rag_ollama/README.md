# Knowledge Graph RAG System with Ollama

A comprehensive Knowledge Graph-based Retrieval-Augmented Generation (RAG) system that combines multiple approaches for extracting, analyzing, and querying knowledge from documents. This project features a modern **Streamlit web interface**, modular architecture, and advanced visualization capabilities.

## 🎯 Project Overview

This repository contains three main interfaces:

1. **🌐 Streamlit Web App** (`app.py`) - **Modern web interface** with interactive visualizations, document upload, and real-time analytics
2. **💻 Enhanced CLI System** (`main.py`) - Comprehensive command-line interface with all advanced features  
3. **🎮 Demo Script** (`demo.py`) - Lightweight demonstration of basic knowledge graph extraction

## ✨ **New in v2.0: Modular Architecture & Web Interface**

- **🌐 Modern Streamlit Web UI** - Intuitive interface with drag-drop file upload
- **🏗️ Modular Codebase** - Clean separation into 8 core modules  
- **📊 Interactive Visualizations** - Real-time Plotly charts and dashboards
- **🔄 Session State Management** - Persistent data across interactions
- **📁 Multiple Document Formats** - PDF, TXT, MD, DOCX support
- **💾 Export Capabilities** - JSON, CSV, analytics reports

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│            Knowledge Graph RAG System - Workflow Architecture          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │                      1. USER INTERFACES                            │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐                   │
│  │  │ Streamlit  │   │    CLI     │   │    Demo    │                   │
│  │  │  Web App   │   │ Interface  │   │   Script   │                   │
│  │  │   (app.py) │   │ (main.py)  │   │ (demo.py)  │                   │
│  │  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                   │
│  └────────┼────────────────┼────────────────┼────────────────────────────┘
│           │                │                │                          │
│           └────────────────┼────────────────┘                          │
│                            ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │                2. CONFIGURATION & LLM SETUP                        │
│  │                                                                     │
│  │  ┌─────────────┐    ┌──────────────────────────────────────────┐    │
│  │  │   Models    │───▶│            OLLAMA SERVER                │    │
│  │  │ (models.py) │    │         (localhost:11434)               │    │
│  │  │             │    │                                          │    │
│  │  │• Config     │    │  ┌─────────────┐  ┌─────────────────┐   │    │
│  │  │• Filters    │    │  │ LLM Models  │  │ Embedding Models│   │    │
│  │  │• QueryTypes │    │  │             │  │                 │   │    │
│  │  └─────────────┘    │  │• llama3.2   │  │• nomic-embed-   │   │    │
│           │             │  │• llama3.1   │  │  text           │   │    │
│           │             │  │• codellama  │  │• mxbai-embed-   │   │    │
│           │             │  └─────────────┘  │  large          │   │    │
│           │             └──────────────────┴─────────────────────┘    │
│           │                              ▲                           │
│           │             ┌────────────────┴─────────────────┐          │
│           │             │       LLM Setup (llm_setup.py)  │          │
│           │             │                                  │          │
│           │             │ • Ollama LLM Integration        │          │
│           │             │ • Embedding Model Setup        │          │
│           │             │ • Connection Testing            │          │
│           │             │ • Model Configuration           │          │
│           │             └──────────────────────────────────┘          │
│           ▼                              │                           │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │               3. DOCUMENT PROCESSING & STORAGE                     │
│  │                                                                     │
│  │  ┌─────────────┐     ┌─────────────┐      ┌──────────────────────┐ │
│  │  │ Documents   │────▶│   Storage   │─────▶│    Document Parsing  │ │
│  │  │   Input     │     │ Manager     │      │                      │ │
│  │  │             │     │(storage.py) │      │ • SentenceSplitter   │ │
│  │  │• PDF, TXT   │     │             │      │ • Chunk Creation     │ │
│  │  │• MD, DOCX   │     │• File Load  │      │ • Metadata Extract   │ │
│  │  │• Directories│     │• Format     │      └──────────────────────┘ │
│  │  └─────────────┘     │• Validation │               │               │
│  │                      └─────────────┘               ▼               │
│  └────────────────────────────────────────────────────────────────────┘
│                                                       │               │
│                                                       ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │              4. KNOWLEDGE EXTRACTION & GRAPH BUILDING              │
│  │                                                                     │
│  │  ┌──────────────────────┐       ┌─────────────────────────────────┐ │
│  │  │   Graph Builder      │◄─────▶│         OLLAMA PROCESSING       │ │
│  │  │  (graph_builder.py)  │       │                                 │ │
│  │  │                      │       │  ┌─────────────────────────────┐ │ │
│  │  │ • KG Index Creation  │       │  │     Entity Extraction       │ │ │
│  │  │ • Vector Index Build │       │  │   (Subject-Predicate-Obj)   │ │ │
│  │  │ • NetworkX Graph     │       │  │                             │ │ │
│  │  │ • Triplet Extraction │       │  │ • Named Entity Recognition  │ │ │
│  │  └──────────────────────┘       │  │ • Relationship Detection    │ │ │
│  │              │                  │  │ • Confidence Scoring        │ │ │
│  │              ▼                  │  └─────────────────────────────┘ │ │
│  │  ┌──────────────────────┐       │  ┌─────────────────────────────┐ │ │
│  │  │    Storage Layer     │       │  │      Text Embeddings        │ │ │
│  │  │                      │       │  │                             │ │ │
│  │  │ • Knowledge Graph    │       │  │ • Semantic Vector Creation  │ │ │
│  │  │   (enhanced_kg_      │       │  │ • Similarity Computation    │ │ │
│  │  │   storage/)          │       │  │ • Clustering & Indexing     │ │ │
│  │  │ • Vector Database    │       │  └─────────────────────────────┘ │ │
│  │  │   (Chroma/Qdrant)    │       └─────────────────────────────────┘ │
│  │  └──────────────────────┘                                         │
│  └─────────────────────────────────────────────────────────────────────┘
│                                     │                                 │
│                                     ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │                5. QUERY PROCESSING & RETRIEVAL                     │
│  │                                                                     │
│  │  ┌─────────────┐    ┌──────────────────────────────────────────┐   │
│  │  │   Query     │───▶│         HYBRID RETRIEVAL                 │   │
│  │  │  Engine     │    │                                          │   │
│  │  │(query_      │    │  ┌─────────────┐  ┌──────────────────┐   │   │
│  │  │engine.py)   │    │  │Knowledge    │  │   Vector Search  │   │   │
│  │  │             │    │  │Graph Query  │  │                  │   │   │
│  │  │• Question   │    │  │             │  │ • Semantic       │   │   │
│  │  │  Processing │    │  │• Triplet    │  │   Similarity     │   │   │
│  │  │• Filter     │    │  │  Matching   │  │ • Embedding      │   │   │
│  │  │  Application│    │  │• Path       │  │   Search         │   │   │
│  │  │• Response   │    │  │  Finding    │  │ • Top-K Results  │   │   │
│  │  │  Generation │    │  │• Entity     │  │                  │   │   │
│  │  └─────────────┘    │  │  Relations  │  │                  │   │   │
│  │         │            │  └─────────────┘  └──────────────────┘   │   │
│  │         │            └────────────────┬─────────────────────────┘   │
│  │         │                             │                             │
│  │         │            ┌────────────────▼─────────────────┐           │
│  │         │            │         OLLAMA LLM               │           │
│  │         │            │                                  │           │
│  │         │            │ • Context Understanding         │           │
│  │         │            │ • Response Generation           │           │
│  │         │            │ • Answer Synthesis              │           │
│  │         │            │ • Source Attribution            │           │
│  │         │            └──────────────────────────────────┘           │
│  │         ▼                                                           │
│  └─────────────────────────────────────────────────────────────────────┘
│                                     │                                 │
│                                     ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │              6. ANALYTICS & VISUALIZATION                          │
│  │                                                                     │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  │ Analytics   │    │Visualization│    │   Export    │             │
│  │  │(analytics.py│    │(visualization    │(export_utils│             │
│  │  │             │    │.py)         │    │.py)         │             │
│  │  │• Graph      │    │             │    │             │             │
│  │  │  Metrics    │    │• Interactive│    │• JSON       │             │
│  │  │• Centrality │    │  Plotly     │    │• CSV        │             │
│  │  │• Communities│    │• NetworkX   │    │• GEXF       │             │
│  │  │• Statistics │    │• Dashboards │    │• Reports    │             │
│  │  └─────────────┘    └─────────────┘    └─────────────┘             │
│  └─────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────┘

WORKFLOW:
User Input → Config/LLM Setup → Document Processing → Ollama Extraction → 
Graph Building → Storage → Query Processing → Ollama Response → Analytics → Visualization
```

## ✨ Features

### Core Features
- **Knowledge Graph Construction**: Extract entities and relationships from text documents
- **Interactive Query System**: Natural language querying with hybrid retrieval
- **Graph Visualization**: Interactive visualizations using Plotly and NetworkX
- **Analytics Dashboard**: Comprehensive graph metrics and community detection
- **Multiple Vector Databases**: Support for both Chroma and Qdrant
- **Persistent Storage**: Save and reload knowledge graphs between sessions

### Advanced Capabilities
- **Hybrid Retrieval**: Combines knowledge graph and vector search
- **Advanced Filtering**: Filter results by confidence, entities, relations, and dates
- **Knowledge Path Discovery**: Find connections between entities
- **Community Detection**: Automatic clustering of related entities
- **Export Functionality**: Export graphs in JSON, CSV, and other formats

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required Python packages (see requirements.txt)

### Setup Instructions
```bash
# Clone the repository
git clone <repository-url>
cd graph_rag_ollama

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Start Ollama and pull required models
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

#### Troubleshooting Package Installation
If you encounter issues with package versions, try installing core packages individually:
```bash
# Core LlamaIndex
pip install llama-index

# Ollama integration
pip install llama-index-llms-ollama llama-index-embeddings-ollama

# Vector databases
pip install chromadb qdrant-client

# Visualization
pip install plotly networkx pandas matplotlib streamlit

# NLP
pip install spacy transformers
python -m spacy download en_core_web_sm
```

## 📂 Project Structure

```
graph_rag_ollama/
├── 🌐 User Interfaces
│   ├── app.py                   # Streamlit web application (⭐ RECOMMENDED)
│   ├── main.py                  # Enhanced CLI interface (refactored)
│   └── demo.py                  # Demo script for basic functionality
│
├── 🏗️ Core Modules (Modular Architecture)
│   ├── models.py               # Data models & configuration classes
│   ├── storage.py              # Storage management (graphs & vectors)
│   ├── llm_setup.py            # LLM and embedding model setup
│   ├── graph_builder.py        # Knowledge graph construction
│   ├── query_engine.py         # Query processing & hybrid retrieval
│   ├── analytics.py            # Graph analytics & metrics calculation
│   ├── visualization.py        # Interactive Plotly visualizations
│   └── export_utils.py         # Data export utilities (JSON, CSV)
│
├── 📁 Data & Storage
│   ├── data/                   # Input documents directory
│   │   └── sample.txt          # Sample text data
│   ├── enhanced_kg_storage/    # Persistent knowledge graph storage
│   │   ├── docstore.json
│   │   ├── graph_store.json
│   │   ├── index_store.json
│   │   └── image__vector_store.json
│   └── vector_db/              # Vector database storage (Chroma)
│       ├── chroma.sqlite3
│       └── [additional vector files]
│
├── ⚙️ Configuration
│   ├── requirements.txt        # Python dependencies
│   └── .venv/                  # Virtual environment
```

## 🚀 Quick Start

### 🌐 **Web Interface (⭐ Recommended)**

**The modern way to use the system:**

```bash
streamlit run app.py
```

**🎯 Web Interface Features:**
- **📁 Drag & Drop Upload** - Easy document upload or directory selection
- **🎨 Interactive Visualizations** - Real-time Plotly charts with zoom/pan
- **📊 Analytics Dashboard** - Comprehensive graph metrics and insights  
- **💬 Query Interface** - Natural language queries with history
- **⚙️ Configuration Panel** - Easy model and database settings
- **💾 Export Tools** - Download results in JSON/CSV formats
- **🔄 Session Persistence** - Data stays loaded across interactions

### 💻 **Command Line Interface**

**For developers and power users:**

```bash
python main.py
```

**CLI Features:**
- Interactive commands with autocomplete
- Advanced filtering and query options
- Built-in help system and command reference
- Export functionality and graph analytics

### 🎮 **Demo Script**

**For basic demonstration and experimentation:**

```bash
python demo.py
```

**Demo Features:**
- Standalone entity and relationship extraction
- Basic graph visualization with matplotlib
- JSON export for external tools
- No dependency on persistent storage
- Perfect for learning and testing concepts

## 🎮 **Using the Web Interface**

### **📊 Getting Started (Web App)**

1. **🚀 Launch the App**
   ```bash
   streamlit run app.py
   ```
   Open your browser to `http://localhost:8501`

2. **📁 Upload Documents**
   - **Drag & drop files** directly into the interface
   - **Or specify a directory path** containing your documents
   - Supported formats: PDF, TXT, MD, DOCX

3. **🔨 Build Knowledge Graph**
   - Click "Build New Graph" 
   - Wait for processing (progress shown)
   - Review generated statistics

4. **💬 Start Querying**
   - Switch to "Query" tab
   - Ask natural language questions
   - View results with sources and related entities

5. **🎨 Explore Visualizations**
   - Interactive graph visualization
   - Analytics dashboard with metrics
   - Customizable layouts and node limits

### **💻 CLI Commands (Terminal Interface)**

```bash
# Basic Querying
query What is machine learning?
advanced How do neural networks work?
<your question>                    # Default to advanced query

# Analysis & Visualization  
visualize                          # Interactive graph visualization
analytics                          # Comprehensive dashboard
stats                             # Basic graph statistics

# Entity & Path Discovery
entities artificial intelligence   # Find related entities
path "entity1" "entity2"          # Shortest path between entities

# Configuration & Export
filter                            # Configure query filters
export                            # Export graph data
help                              # Show all commands
quit                              # Exit system
```

## ⚙️ **Configuration & Settings**

### **🌐 Web Interface Configuration**

Configure everything through the **sidebar** in the Streamlit app:

- **🤖 LLM Model**: Choose from llama3.2, llama3.1:8b, codellama
- **🔤 Embedding Model**: nomic-embed-text, mxbai-embed-large, all-minilm  
- **🗄️ Vector Database**: Chroma (lightweight) or Qdrant (production)
- **🔧 Advanced Settings**: Triplets per chunk, chunk size, Ollama URL

### **💻 CLI Configuration**

```python
# Programmatic configuration in main.py
config = KnowledgeGraphConfig(
    model_name='llama3.2',              # LLM model
    embedding_model='nomic-embed-text', # Embedding model  
    vector_db_type='chroma',            # Vector database
    max_triplets_per_chunk=10,          # Knowledge extraction
    chunk_size=512,                     # Text processing
    ollama_url='http://localhost:11434' # Ollama server
)
```

### **🗄️ Database Comparison**

| Feature | **Chroma** | **Qdrant** |
|---------|------------|-------------|
| **Use Case** | Development, Prototyping | Production, Scale |
| **Setup** | Zero-config | Configurable |
| **Performance** | Good for <100K docs | Excellent for millions |
| **Memory** | Lower | Higher but optimized |
| **Features** | Basic | Advanced filtering |

## 📊 Graph Analytics

The system provides comprehensive graph analytics including:

- **Network Metrics**: Node count, edge count, density, clustering coefficient
- **Centrality Measures**: Betweenness, closeness, PageRank centrality
- **Community Detection**: Automatic clustering of related entities
- **Degree Distribution**: Understanding connectivity patterns
- **Export Capabilities**: JSON, CSV formats for external analysis

## 🧪 Example Usage

### Sample Text Processing
The system can process various document types (PDF, TXT, MD, DOCX) and extract knowledge graphs. Here's an example with the included sample data:

**Input Text:**
```
Alice works for Acme Corp. Charlie founded Acme Corp. Alice reports to Charlie.
Bob works for Globex. Alice knows Bob.
Globex acquired Acme Corp. Bob reports to Charlie.
```

**Extracted Knowledge Graph:**
- Entities: Alice, Bob, Charlie, Acme Corp, Globex
- Relationships: works_for, founded, reports_to, knows, acquired

### Query Examples
```bash
# Basic information retrieval
"Who works for Acme Corp?"

# Relationship discovery
"What is the connection between Alice and Bob?"

# Complex queries
"How are the companies related in this network?"
```

## 📦 **Dependencies & Requirements**

### **🧠 Core AI Stack**
- **llama-index>=0.10.0** - LLM framework and indexing
- **llama-index-llms-ollama** - Ollama LLM integration  
- **llama-index-embeddings-ollama** - Ollama embedding models
- **ollama** - Local LLM server (install separately)

### **🗄️ Vector Databases**  
- **chromadb>=0.4.0** - Lightweight vector database
- **qdrant-client>=1.6.0** - Production vector database
- **scipy>=1.11.0** - Scientific computing for analytics

### **🎨 Visualization & Web Interface**
- **streamlit>=1.28.0** - Modern web interface
- **plotly>=5.17.0** - Interactive visualizations  
- **networkx>=3.2.0** - Graph analysis and algorithms
- **matplotlib>=3.7.0** - Static graph plotting
- **kaleido>=0.2.1** - Plot export functionality

### **🔤 NLP & Data Processing**
- **spacy>=3.7.0** - Advanced NLP processing
- **transformers>=4.35.0** - Transformer models
- **pandas>=2.1.0** - Data manipulation
- **requests>=2.31.0** - HTTP client for APIs

### **🚀 Quick Install**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 🛠️ **Troubleshooting**

### **🔧 Common Issues & Solutions**

#### **🤖 Ollama Connection Problems**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server
ollama serve

# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

#### **💾 Memory Issues**
- **Reduce processing load**: Lower `max_triplets_per_chunk` to 5-8
- **Use smaller models**: Try `llama3.2:1b` for development
- **Limit visualizations**: Set node limit to 30-50 for large graphs
- **Chunk size**: Reduce to 256 for memory-constrained systems

#### **📦 Package Installation Issues**
```bash
# Missing spaCy model
python -m spacy download en_core_web_sm

# Visualization dependencies
pip install kaleido scipy

# If LlamaIndex packages fail
pip install llama-index chromadb plotly streamlit
```

#### **🌐 Streamlit Issues**
- **App won't start**: Check all dependencies installed
- **Stuck loading**: Clear browser cache, restart app
- **Upload fails**: Check file permissions and formats
- **Visualization errors**: Install scipy and kaleido

#### **⚡ Performance Optimization**
```bash
# For better performance
pip install watchdog              # File watching
xcode-select --install           # macOS development tools (optional)

# Use faster models
ollama pull llama3.2:1b          # Smaller, faster model
ollama pull all-minilm           # Faster embeddings
```

## 🚀 **Performance Tuning**

### **📈 For Large Document Collections**
- **Vector Database**: Use Qdrant for >100K documents  
- **Chunk Processing**: Increase chunk_size to 1024 for technical docs
- **Persistence**: Enable storage to reuse built indices
- **Batch Processing**: Process documents in smaller batches

### **🎯 For Better Accuracy**
- **Larger Models**: Use llama3.1:8b or codellama for better quality
- **Better Embeddings**: Switch to mxbai-embed-large
- **More Relationships**: Increase max_triplets_per_chunk to 15-20
- **Fine-tuning**: Adjust confidence thresholds based on your domain

### **⚡ For Better Speed**  
- **Smaller Models**: Use llama3.2:1b for development
- **Reduced Triplets**: Lower max_triplets_per_chunk to 5-8
- **Node Limits**: Limit visualization to 30-50 nodes
- **Local Storage**: Use SSD for vector database storage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🏗️ New Modular Architecture

The project has been refactored into a clean, modular architecture:

### 📋 Core Modules

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `models.py` | Data models & config | QueryFilter, GraphMetrics, KnowledgeGraphConfig |
| `storage.py` | Storage management | Document loading, index persistence, vector DB setup |
| `llm_setup.py` | LLM configuration | Ollama integration, model management |
| `graph_builder.py` | Graph construction | Knowledge graph building, NetworkX integration |
| `query_engine.py` | Query processing | Hybrid retrieval, advanced filtering |
| `analytics.py` | Graph analytics | Metrics calculation, centrality analysis |
| `visualization.py` | Graph visualization | Plotly charts, interactive dashboards |
| `export_utils.py` | Data export | JSON, CSV, GEXF export formats |

### 🌐 User Interfaces

- **`app.py`**: Modern Streamlit web application with intuitive UI
- **`main.py`**: Refactored CLI using the modular architecture
- **`demo.py`**: Standalone demonstration script for basic functionality

### ✨ Benefits of New Architecture

- **🔧 Maintainable**: Clean separation of concerns
- **🔗 Extensible**: Easy to add new features and modules
- **🧪 Testable**: Each module can be tested independently
- **📚 Reusable**: Components can be used in other projects
- **🌐 Web-Ready**: Streamlit app provides modern interface

## 🔮 **Future Roadmap**

### **🎯 Planned Features (v2.1)**
- **📱 Mobile-responsive web interface** - Better mobile experience
- **🔄 Real-time document monitoring** - Auto-update graphs when files change
- **🧪 A/B testing framework** - Compare different extraction strategies
- **📊 Advanced analytics** - More sophisticated graph metrics

### **🚀 Advanced Features (v3.0)**
- **🖼️ Multi-modal support** - Images, tables, charts integration
- **🌐 REST API interface** - For external application integration  
- **🤝 Collaborative features** - Multi-user knowledge graphs
- **🗄️ Graph database backends** - Neo4j, ArangoDB integration
- **🧠 Advanced NLP models** - Better entity and relation extraction

### **🏗️ Infrastructure (v3.1)**
- **🧪 Automated testing suite** - Comprehensive test coverage
- **🔄 CI/CD pipeline** - Automated deployment and testing
- **📊 Monitoring & observability** - Performance and usage metrics
- **🔒 Authentication & security** - User management and access control

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify Ollama is running with required models
4. Create an issue in the repository for bugs or feature requests

---

**Note**: This project is designed for educational and research purposes. Ensure you have appropriate permissions to process any documents you use with the system.