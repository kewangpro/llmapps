"""
Streamlit Web Application for Knowledge Graph RAG System.
"""

import streamlit as st
import logging
import json
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Import our modular components
from models import KnowledgeGraphConfig, QueryFilter, QueryResult
from storage import StorageManager
from llm_setup import LLMManager
from graph_builder import GraphBuilder
from query_engine import QueryEngine
from analytics import GraphAnalytics
from visualization import GraphVisualizer
from export_utils import GraphExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Knowledge Graph RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class KnowledgeGraphApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.config = None
        self.storage_manager = None
        self.llm_manager = None
        self.graph_builder = None
        self.query_engine = None
        self.analytics = None
        self.visualizer = None
        self.exporter = None
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'kg_index' not in st.session_state:
            st.session_state.kg_index = None
        if 'vector_index' not in st.session_state:
            st.session_state.vector_index = None
        if 'networkx_graph' not in st.session_state:
            st.session_state.networkx_graph = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'current_config' not in st.session_state:
            st.session_state.current_config = None
        if 'ollama_tested' not in st.session_state:
            st.session_state.ollama_tested = False
        if 'storage_manager' not in st.session_state:
            st.session_state.storage_manager = None
        if 'llm_manager' not in st.session_state:
            st.session_state.llm_manager = None
        if 'graph_builder' not in st.session_state:
            st.session_state.graph_builder = None
        if 'query_engine' not in st.session_state:
            st.session_state.query_engine = None
    
    def setup_sidebar(self):
        """Setup the sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        # Model Configuration
        st.sidebar.subheader("🤖 Model Settings")
        model_name = st.sidebar.selectbox(
            "LLM Model",
            ["llama3.2", "llama3.1:8b", "llama3.2:1b", "codellama"],
            index=0
        )
        
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
            index=0
        )
        
        # Vector Database Configuration
        st.sidebar.subheader("🗄️ Vector Database")
        vector_db_type = st.sidebar.selectbox(
            "Database Type",
            ["chroma", "qdrant"],
            index=0
        )
        
        # Advanced Settings
        st.sidebar.subheader("🔧 Advanced Settings")
        max_triplets = st.sidebar.slider(
            "Max Triplets per Chunk",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of knowledge triplets to extract per text chunk"
        )
        
        chunk_size = st.sidebar.slider(
            "Chunk Size",
            min_value=256,
            max_value=1024,
            value=512,
            help="Size of text chunks for processing"
        )
        
        # Ollama URL
        ollama_url = st.sidebar.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="URL for Ollama API"
        )
        
        # Create configuration
        config = KnowledgeGraphConfig(
            model_name=model_name,
            embedding_model=embedding_model,
            ollama_url=ollama_url,
            vector_db_type=vector_db_type,
            max_triplets_per_chunk=max_triplets,
            chunk_size=chunk_size
        )
        
        return config
    
    def initialize_system(self, config: KnowledgeGraphConfig):
        """Initialize the knowledge graph system."""
        try:
            # Check if system is already initialized with the same config
            if (st.session_state.system_initialized and 
                st.session_state.current_config and
                st.session_state.current_config.to_dict() == config.to_dict()):
                # System already initialized with same config, reuse components
                return True
            
            with st.spinner("🔄 Initializing system..."):
                self.config = config
                st.session_state.current_config = config
                
                # Store components in session state to avoid reinitializing
                prev_config = getattr(st.session_state, 'previous_config', None)
                config_changed = (prev_config is None or 
                                prev_config.to_dict() != config.to_dict())
                
                if 'storage_manager' not in st.session_state or config_changed:
                    st.session_state.previous_config = config
                    st.session_state.storage_manager = StorageManager(config)
                    st.session_state.llm_manager = LLMManager(config)
                    st.session_state.graph_builder = GraphBuilder(config, st.session_state.storage_manager)
                    st.session_state.query_engine = QueryEngine(config)
                
                # Use session state components
                self.storage_manager = st.session_state.storage_manager
                self.llm_manager = st.session_state.llm_manager
                self.graph_builder = st.session_state.graph_builder
                self.query_engine = st.session_state.query_engine
                
                # Test Ollama connection only if not already tested
                if 'ollama_tested' not in st.session_state:
                    if not self.llm_manager.test_connection():
                        st.error("❌ Failed to connect to Ollama. Please ensure Ollama is running and the models are installed.")
                        return False
                    st.session_state.ollama_tested = True
                
                st.session_state.system_initialized = True
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {e}")
            logger.error(f"System initialization error: {e}")
            return False
    
    def load_or_build_knowledge_graph(self):
        """Load existing knowledge graph or build new one."""
        st.subheader("📚 Knowledge Graph Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Load Existing Graph", help="Load previously built knowledge graph"):
                with st.spinner("Loading existing knowledge graph..."):
                    kg_index, vector_index = None, None
                    if st.session_state.storage_manager:
                        kg_index, vector_index = st.session_state.storage_manager.load_existing_indices()
                    
                    if kg_index:
                        st.session_state.kg_index = kg_index
                        st.session_state.vector_index = vector_index
                        
                        # Rebuild NetworkX graph
                        if st.session_state.graph_builder:
                            st.session_state.graph_builder.rebuild_networkx_graph(kg_index)
                            st.session_state.networkx_graph = st.session_state.graph_builder.get_networkx_graph()
                        
                        # Setup query engines
                        if st.session_state.query_engine:
                            st.session_state.query_engine.setup_query_engines(
                                kg_index, vector_index, st.session_state.networkx_graph
                            )
                        
                        # Setup analytics and visualization
                        self.analytics = GraphAnalytics(st.session_state.networkx_graph)
                        self.visualizer = GraphVisualizer(st.session_state.networkx_graph)
                        self.exporter = GraphExporter(st.session_state.networkx_graph)
                        
                        st.success("✅ Knowledge graph loaded successfully!")
                        self._display_graph_stats()
                    else:
                        st.warning("⚠️ No existing knowledge graph found.")
        
        with col2:
            if st.button("🔨 Build New Graph", help="Build knowledge graph from documents"):
                self._build_new_graph_interface()
    
    def _build_new_graph_interface(self):
        """Interface for building new knowledge graph."""
        st.subheader("📁 Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'md', 'docx'],
            accept_multiple_files=True,
            help="Upload documents to build knowledge graph from"
        )
        
        # Directory path input as alternative
        data_dir = st.text_input(
            "Or provide directory path",
            placeholder="/path/to/documents",
            help="Alternative: provide path to directory containing documents"
        )
        
        if st.button("🚀 Build Knowledge Graph"):
            if uploaded_files or (data_dir and Path(data_dir).exists()):
                self._build_knowledge_graph(uploaded_files, data_dir)
            else:
                st.error("❌ Please upload files or provide a valid directory path.")
    
    def _build_knowledge_graph(self, uploaded_files, data_dir):
        """Build knowledge graph from uploaded files or directory."""
        try:
            with st.spinner("🔨 Building knowledge graph... This may take a few minutes."):
                # Handle uploaded files
                if uploaded_files:
                    # Create temporary directory
                    temp_dir = tempfile.mkdtemp()
                    
                    for uploaded_file in uploaded_files:
                        file_path = Path(temp_dir) / uploaded_file.name
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                    
                    documents = st.session_state.storage_manager.load_documents(temp_dir) if st.session_state.storage_manager else []
                else:
                    # Use provided directory
                    documents = st.session_state.storage_manager.load_documents(data_dir) if st.session_state.storage_manager else []
                
                if not documents:
                    st.error("❌ No documents found or failed to load documents.")
                    return
                
                st.info(f"📄 Loaded {len(documents)} documents")
                
                # Build knowledge graph
                if st.session_state.graph_builder:
                    kg_index, vector_index = st.session_state.graph_builder.build_knowledge_graph(
                        documents, 
                        include_embeddings=True,
                        build_vector_index=True
                    )
                    
                    # Store in session state
                    st.session_state.kg_index = kg_index
                    st.session_state.vector_index = vector_index
                    st.session_state.networkx_graph = st.session_state.graph_builder.get_networkx_graph()
                    
                    # Setup query engines
                    if st.session_state.query_engine:
                        st.session_state.query_engine.setup_query_engines(
                            kg_index, vector_index, st.session_state.networkx_graph
                        )
                    
                    # Setup analytics and visualization
                    self.analytics = GraphAnalytics(st.session_state.networkx_graph)
                    self.visualizer = GraphVisualizer(st.session_state.networkx_graph)
                    self.exporter = GraphExporter(st.session_state.networkx_graph)
                
                st.success("✅ Knowledge graph built successfully!")
                self._display_graph_stats()
                
                # Clean up temporary directory
                if uploaded_files:
                    import shutil
                    shutil.rmtree(temp_dir)
                
        except Exception as e:
            st.error(f"❌ Failed to build knowledge graph: {e}")
            logger.error(f"Graph building error: {e}")
    
    def _display_graph_stats(self):
        """Display basic graph statistics."""
        if self.analytics:
            metrics = self.analytics.get_graph_metrics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="🔗 Entities",
                    value=metrics.node_count
                )
            
            with col2:
                st.metric(
                    label="↔️ Relationships",
                    value=metrics.edge_count
                )
            
            with col3:
                st.metric(
                    label="🏘️ Communities",
                    value=len(metrics.communities)
                )
            
            with col4:
                st.metric(
                    label="📊 Density",
                    value=f"{metrics.density:.3f}"
                )
    
    def query_interface(self):
        """Query interface for the knowledge graph."""
        if not st.session_state.kg_index:
            st.warning("⚠️ Please load or build a knowledge graph first.")
            return
        
        st.subheader("💬 Query Knowledge Graph")
        
        # Query input
        query_text = st.text_area(
            "Ask a question",
            placeholder="What is the relationship between machine learning and artificial intelligence?",
            height=100
        )
        
        # Query options
        col1, col2 = st.columns(2)
        
        with col1:
            use_hybrid = st.checkbox(
                "Use Hybrid Search",
                value=True,
                help="Combine knowledge graph and vector search"
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Filter results by confidence score"
            )
        
        if st.button("🔍 Search") and query_text:
            self._process_query(query_text, use_hybrid, min_confidence)
    
    def _process_query(self, query_text: str, use_hybrid: bool, min_confidence: float):
        """Process a query and display results."""
        try:
            with st.spinner("🔍 Processing query..."):
                # Create query filter
                query_filter = QueryFilter(min_confidence=min_confidence)
                
                # Execute query
                result = None
                if st.session_state.query_engine:
                    result = st.session_state.query_engine.advanced_query(
                        query_text, 
                        query_filter=query_filter,
                        use_hybrid=use_hybrid
                    )
                
                if result:
                    # Add to query history
                    st.session_state.query_history.append({
                        'timestamp': datetime.now(),
                        'query': query_text,
                        'result': result
                    })
                    
                    # Display results
                    self._display_query_result(result)
                else:
                    st.error("❌ Query engine not available or query failed")
                
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
            logger.error(f"Query processing error: {e}")
    
    def _display_query_result(self, result: QueryResult):
        """Display query results."""
        st.subheader("🤖 Answer")
        st.write(result.answer)
        
        # Related entities
        if result.related_entities:
            st.subheader("🔗 Related Entities")
            st.write(", ".join(result.related_entities[:10]))
        
        # Knowledge paths
        if result.knowledge_paths:
            st.subheader("🛤️ Knowledge Paths")
            for i, path in enumerate(result.knowledge_paths[:3], 1):
                st.write(f"{i}. {' → '.join(path)}")
        
        # Sources
        if result.sources:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(result.sources[:5], 1):
                    st.write(f"**{i}. [{source['source_type'].upper()}]** (Score: {source['score']:.3f})")
                    st.write(source['text'])
                    st.write("---")
    
    def visualization_interface(self):
        """Interface for graph visualization."""
        if not st.session_state.networkx_graph:
            st.warning("⚠️ Please load or build a knowledge graph first.")
            return
        
        st.subheader("🎨 Graph Visualization")
        
        # Visualization options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            layout = st.selectbox(
                "Layout Algorithm",
                ["spring", "circular", "random"],
                index=0
            )
        
        with col2:
            node_limit = st.slider(
                "Max Nodes",
                min_value=10,
                max_value=200,
                value=50,
                help="Limit nodes for performance"
            )
        
        with col3:
            show_labels = st.checkbox(
                "Show Labels",
                value=True
            )
        
        if st.button("🎨 Generate Visualization"):
            with st.spinner("Creating visualization..."):
                try:
                    # Initialize visualizer if not available
                    if not hasattr(self, 'visualizer') or self.visualizer is None:
                        self.visualizer = GraphVisualizer(st.session_state.networkx_graph)
                    
                    fig = self.visualizer.visualize_graph(
                        layout=layout,
                        node_limit=node_limit,
                        show_labels=show_labels
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Visualization failed: {e}")
                    logger.error(f"Visualization error: {e}")
    
    def analytics_interface(self):
        """Interface for graph analytics."""
        if not st.session_state.networkx_graph:
            st.warning("⚠️ Please load or build a knowledge graph first.")
            return
        
        st.subheader("📊 Graph Analytics")
        
        # Initialize analytics if not available
        if not hasattr(self, 'analytics') or self.analytics is None:
            self.analytics = GraphAnalytics(st.session_state.networkx_graph)
        
        # Display metrics
        metrics = self.analytics.get_graph_metrics()
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Graph Overview</h4>
                <p>Entities: {}</p>
                <p>Relationships: {}</p>
                <p>Density: {:.3f}</p>
            </div>
            """.format(metrics.node_count, metrics.edge_count, metrics.density), 
            unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Network Properties</h4>
                <p>Avg Degree: {:.2f}</p>
                <p>Clustering: {:.3f}</p>
                <p>Communities: {}</p>
            </div>
            """.format(metrics.avg_degree, metrics.clustering_coefficient, len(metrics.communities)), 
            unsafe_allow_html=True)
        
        with col3:
            if metrics.centrality_scores.get('pagerank'):
                top_entity = max(metrics.centrality_scores['pagerank'].items(), key=lambda x: x[1])
                st.markdown("""
                <div class="metric-card">
                    <h4>Most Central Entity</h4>
                    <p>{}</p>
                    <p>PageRank: {:.3f}</p>
                </div>
                """.format(top_entity[0], top_entity[1]), 
                unsafe_allow_html=True)
        
        # Analytics dashboard
        if st.button("📊 Show Analytics Dashboard"):
            with st.spinner("Creating analytics dashboard..."):
                try:
                    # Initialize visualizer if not available
                    if not hasattr(self, 'visualizer') or self.visualizer is None:
                        self.visualizer = GraphVisualizer(st.session_state.networkx_graph)
                    
                    dashboard = self.visualizer.create_analytics_dashboard()
                    st.plotly_chart(dashboard, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Dashboard creation failed: {e}")
                    logger.error(f"Dashboard creation error: {e}")
    
    def export_interface(self):
        """Interface for exporting graph data."""
        if not st.session_state.networkx_graph:
            st.warning("⚠️ Please load or build a knowledge graph first.")
            return
        
        st.subheader("💾 Export Data")
        
        # Initialize exporter if not available
        if not hasattr(self, 'exporter') or self.exporter is None:
            self.exporter = GraphExporter(st.session_state.networkx_graph)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export to JSON"):
                try:
                    json_data = self.exporter.export_to_json(None)
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_data,
                        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"❌ JSON export failed: {e}")
                    logger.error(f"JSON export error: {e}")
        
        with col2:
            if st.button("📊 Export Analytics"):
                try:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    temp_file = f"/tmp/centrality_report_{timestamp}.json"
                    self.exporter.export_centrality_report(temp_file)
                    
                    with open(temp_file, 'r') as f:
                        report_data = f.read()
                    
                    st.download_button(
                        label="⬇️ Download Report",
                        data=report_data,
                        file_name=f"analytics_report_{timestamp}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"❌ Analytics export failed: {e}")
                    logger.error(f"Analytics export error: {e}")
    
    def query_history_interface(self):
        """Interface for viewing query history."""
        if not st.session_state.query_history:
            st.info("📝 No queries yet. Start by asking a question!")
            return
        
        st.subheader("📝 Query History")
        
        for i, entry in enumerate(reversed(st.session_state.query_history[-10:]), 1):
            with st.expander(f"Query {i}: {entry['query'][:50]}..."):
                st.write(f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Question:** {entry['query']}")
                st.write(f"**Answer:** {entry['result'].answer}")
    
    def run(self):
        """Main application runner."""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Knowledge Graph RAG System</h1>
            <p>Advanced Knowledge Graph-based Retrieval-Augmented Generation with Ollama</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        config = self.setup_sidebar()
        
        # Initialize system if needed
        if not st.session_state.system_initialized:
            self.initialize_system(config)
        elif (st.session_state.current_config and 
              st.session_state.current_config.to_dict() != config.to_dict()):
            # Only reinitialize if config actually changed
            self.initialize_system(config)
        
        # Main interface tabs
        if st.session_state.system_initialized:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📚 Knowledge Graph", 
                "💬 Query", 
                "🎨 Visualization", 
                "📊 Analytics", 
                "💾 Export",
                "📝 History"
            ])
            
            with tab1:
                self.load_or_build_knowledge_graph()
            
            with tab2:
                self.query_interface()
            
            with tab3:
                self.visualization_interface()
            
            with tab4:
                self.analytics_interface()
            
            with tab5:
                self.export_interface()
            
            with tab6:
                self.query_history_interface()
        
        else:
            st.info("🔧 Please configure the system in the sidebar and ensure Ollama is running.")


def main():
    """Main function to run the Streamlit app."""
    app = KnowledgeGraphApp()
    app.run()


if __name__ == "__main__":
    main()
