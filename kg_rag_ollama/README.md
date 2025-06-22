# Enhanced Knowledge Graph RAG System with Ollama

This project implements an advanced property graph-based RAG system with graph visualization, advanced querying capabilities, and vector database integration using LlamaIndex and Ollama.

## 🚀 New Features

### Graph Visualization
- **Interactive visualizations** using Plotly
- **Network analysis** with NetworkX
- **Analytics dashboard** with comprehensive metrics
- **Community detection** and centrality analysis

### Advanced Querying
- **Hybrid retrieval** combining knowledge graph and vector search
- **Advanced filtering** by confidence, entities, relations, dates
- **Knowledge path discovery** between entities
- **Related entity suggestions**

### Vector Database Integration
- **Chroma** vector database support
- **Qdrant** vector database support (alternative)
- **Persistent storage** for both graph and vector data
- **Hybrid search capabilities**

## Requirements

### Python Dependencies
```bash
# Core LlamaIndex components
pip install llama-index
pip install llama-index-llms-ollama
pip install llama-index-embeddings-ollama
pip install llama-index-graph-stores-simple

# Vector database support
pip install llama-index-vector-stores-chroma
pip install llama-index-vector-stores-qdrant
pip install chromadb
pip install qdrant-client

# Visualization and analytics
pip install plotly
pip install networkx
pip install pandas
pip install kaleido  # For static image export
```

### Ollama Setup
1. Install Ollama from https://ollama.ai/
2. Pull required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
# Optional: Better models for production
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

## Project Structure
```
enhanced-knowledge-graph-rag/
├── main.py                           # Enhanced implementation
├── requirements.txt                   # Python dependencies
├── data/                             # Your documents
│   ├── document1.pdf
│   ├── document2.txt
│   └── research_papers/
├── enhanced_kg_storage/              # Knowledge graph storage
│   ├── index_store.json
│   ├── graph_store.json
│   └── vector_store.json
├── vector_db/                        # Vector database (Chroma/Qdrant)
│   └── chroma.sqlite3
└── exports/                          # Export directory
    ├── graph_20241222_143022.json
    ├── edges_20241222_143022.csv
    └── analytics_20241222_143022.json
```

## 🎮 Interactive Commands

### Basic Querying
```bash
# Standard knowledge graph query
query What is machine learning algorithms?

# Advanced hybrid query (KG + vector search)
advanced How do neural networks work in deep learning?

# Default behavior (same as advanced)
What are the applications of artificial intelligence?
```

### Visualization & Analytics
```bash
# Interactive graph visualization
visualize

# Comprehensive analytics dashboard
analytics

# Graph statistics
stats
```

### Advanced Features
```bash
# Find related entities
entities artificial intelligence

# Discover knowledge paths between entities
path "machine learning" "neural networks"

# Configure query filters
filter

# Export graph data
export
```

## 🛠️ Configuration Options

### Vector Database Selection
```python
# Use Chroma (default, lightweight)
kg_rag = EnhancedKnowledgeGraphRAG(vector_db_type="chroma")

# Use Qdrant (production-ready, scalable)
kg_rag = EnhancedKnowledgeGraphRAG(vector_db_type="qdrant")
```

### Model Configuration
```python
kg_rag = EnhancedKnowledgeGraphRAG(
    model_name="llama3.1:8b",           # Larger model for better quality
    embedding_model="mxbai-embed-large", # Better embeddings
    max_triplets_per_chunk=15,          # More relationships per chunk
    storage_dir="./custom_storage"       # Custom storage location
)
```

### Advanced Query Filtering
```python
from datetime import datetime

query_filter = QueryFilter(
    min_confidence=0.7,                 # Only high-confidence results
    max_hops=3,                        # Deeper graph traversal
    document_sources=["research.pdf"],  # Filter by document
    entity_types=["Person", "Concept"], # Filter by entity types
    date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31))
)

result = kg_rag.advanced_query("your question", query_filter=query_filter)
```

## 📊 Graph Analytics Features

### Network Analysis
- **Centrality measures**: Betweenness, closeness, PageRank
- **Community detection**: Automatic clustering of related entities
- **Degree distribution**: Understanding connectivity patterns
- **Graph density**: Overall connectedness metrics

### Visualization Options
- **Spring layout**: Organic, force-directed positioning
- **Circular layout**: Arranged in a circle
- **Interactive exploration**: Zoom, pan, hover details
- **Entity highlighting**: Focus on specific entities

### Export Capabilities
- **JSON format**: Complete graph structure
- **CSV files**: Nodes and edges for external analysis
- **Analytics data**: Metrics and statistics
- **Visualization**: Interactive HTML files

## 🔧 Performance Optimization

### For Large Documents
```python
# Increase chunk size for longer documents
Settings.chunk_size = 1024

# Reduce triplets per chunk if memory is limited
max_triplets_per_chunk = 5

# Limit visualization nodes for performance
kg_rag.visualize_graph(node_limit=50)
```

### Vector Database Optimization
```python
# Chroma - good for development
vector_db_type = "chroma"

# Qdrant - better for production
vector_db_type = "qdrant"
# Can scale to millions of vectors
# Better query performance
# More configuration options
```

## 📈 Use Cases

### Research & Academia
- **Literature review**: Find connections between research papers
- **Concept mapping**: Understand relationships between ideas
- **Citation analysis**: Track how concepts evolve

### Business Intelligence
- **Document analysis**: Extract insights from company documents
- **Knowledge management**: Create searchable knowledge bases
- **Competitive analysis**: Understand market relationships

### Content Creation
- **Content planning**: Find related topics to cover
- **Fact checking**: Verify relationships between concepts
- **Research assistance**: Quick access to relevant information

## 🐛 Troubleshooting

### Common Issues

#### Ollama Connection
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

#### Memory Issues
```python
# Reduce parameters for lower memory usage
max_triplets_per_chunk = 5
node_limit = 30  # for visualization
Settings.chunk_size = 256
```

#### Slow Performance
```bash
# Use smaller, faster models
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# Or use GPU acceleration
export OLLAMA_NUM_GPU=1
```

#### Graph Visualization Not Showing
```bash
# Install additional dependencies
pip install kaleido nbformat

# For Jupyter notebooks
pip install jupyter plotly
```

### Performance Tips

1. **Start small**: Test with a few documents first
2. **Use appropriate models**: Smaller models for development
3. **Optimize chunk size**: Larger chunks for technical documents
4. **Enable persistence**: Reuse built indices
5. **Filter visualization**: Limit nodes for better performance

## Requirements.txt
```
# Core dependencies
llama-index>=0.9.0
llama-index-llms-ollama>=0.1.0
llama-index-embeddings-ollama>=0.1.0
llama-index-graph-stores-simple>=0.1.0

# Vector databases
llama-index-vector-stores-chroma>=0.1.0
llama-index-vector-stores-qdrant>=0.1.0
chromadb>=0.4.0
qdrant-client>=1.6.0

# Visualization and analytics
plotly>=5.17.0
networkx>=3.2.0
pandas>=2.1.0
kaleido>=0.2.1

# Optional: Enhanced NLP
spacy>=3.7.0
transformers>=4.35.0
```

## 🚦 Getting Started

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull models
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Prepare your documents
mkdir data
# Add your PDF, TXT, MD files to data/

# 4. Run the system
python main.py
```

### First Session
1. **Build the graph**: Point to your documents directory
2. **Explore**: Use `stats` to see what was built
3. **Visualize**: Use `visualize` to see the graph structure
4. **Query**: Ask questions about your documents
5. **Analyze**: Use `analytics` for deeper insights

The system will remember your graph between sessions, so subsequent runs will be much faster!

## 🔮 Future Enhancements

- **Multi-modal support**: Images, tables, charts
- **Real-time updates**: Watch for document changes
- **Advanced NLP**: Better entity and relation extraction
- **Graph databases**: Neo4j, ArangoDB integration
- **Collaborative features**: Multi-user knowledge graphs
- **API interface**: REST API for integration