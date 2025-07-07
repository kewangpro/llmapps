"""
Refactored CLI interface for the Enhanced Knowledge Graph RAG System.
Uses the new modular architecture.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import our modular components
from models import KnowledgeGraphConfig, QueryFilter
from storage import StorageManager
from llm_setup import LLMManager
from graph_builder import GraphBuilder
from query_engine import QueryEngine
from analytics import GraphAnalytics
from visualization import GraphVisualizer
from export_utils import GraphExporter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedKnowledgeGraphRAG:
    """
    Enhanced RAG system with modular architecture.
    """
    
    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        """Initialize the Enhanced Knowledge Graph RAG system."""
        self.config = config or KnowledgeGraphConfig()
        
        # Initialize managers
        self.storage_manager = StorageManager(self.config)
        self.llm_manager = LLMManager(self.config)
        self.graph_builder = GraphBuilder(self.config, self.storage_manager)
        self.query_engine = QueryEngine(self.config)
        
        # Analytics and visualization components
        self.analytics = None
        self.visualizer = None
        self.exporter = None
        
        # Indices
        self.kg_index = None
        self.vector_index = None
        self.networkx_graph = None
    
    def load_documents(self, data_dir: str):
        """Load documents from directory."""
        return self.storage_manager.load_documents(data_dir)
    
    def build_knowledge_graph(self, documents, include_embeddings=True, build_vector_index=True):
        """Build knowledge graph and vector indices."""
        kg_index, vector_index = self.graph_builder.build_knowledge_graph(
            documents, include_embeddings, build_vector_index
        )
        
        self.kg_index = kg_index
        self.vector_index = vector_index
        self.networkx_graph = self.graph_builder.get_networkx_graph()
        
        # Initialize analytics and visualization components
        self._setup_analytics_components()
        
        return kg_index, vector_index
    
    def load_existing_indices(self):
        """Load existing knowledge graph and vector indices."""
        kg_index, vector_index = self.storage_manager.load_existing_indices()
        
        if kg_index:
            self.kg_index = kg_index
            self.vector_index = vector_index
            
            # Rebuild NetworkX graph
            self.graph_builder.rebuild_networkx_graph(kg_index)
            self.networkx_graph = self.graph_builder.get_networkx_graph()
            
            # Initialize analytics and visualization components
            self._setup_analytics_components()
            
            return True
        return False
    
    def _setup_analytics_components(self):
        """Setup analytics, visualization, and export components."""
        if self.networkx_graph:
            self.analytics = GraphAnalytics(self.networkx_graph)
            self.visualizer = GraphVisualizer(self.networkx_graph)
            self.exporter = GraphExporter(self.networkx_graph)
    
    def setup_hybrid_query_engine(self, kg_similarity_top_k=3, vector_similarity_top_k=5, hybrid_weight=0.7):
        """Setup hybrid query engine combining KG and vector retrieval."""
        self.query_engine.setup_query_engines(
            self.kg_index, 
            self.vector_index,
            self.networkx_graph,
            kg_similarity_top_k,
            vector_similarity_top_k,
            hybrid_weight
        )
    
    def advanced_query(self, question: str, query_filter: Optional[QueryFilter] = None, use_hybrid: bool = True):
        """Advanced query with filtering and hybrid retrieval."""
        return self.query_engine.advanced_query(question, query_filter, use_hybrid)
    
    def get_graph_analytics(self):
        """Get comprehensive graph analytics."""
        if self.analytics:
            return self.analytics.get_graph_metrics()
        return None
    
    def visualize_graph(self, layout="spring", node_limit=100, show_labels=True, highlight_entities=None, save_path=None):
        """Create interactive graph visualization using Plotly."""
        if self.visualizer:
            return self.visualizer.visualize_graph(layout, node_limit, show_labels, highlight_entities, save_path)
        return None
    
    def create_analytics_dashboard(self):
        """Create comprehensive analytics dashboard."""
        if self.visualizer:
            return self.visualizer.create_analytics_dashboard()
        return None
    
    def interactive_chat_enhanced(self):
        """Enhanced interactive chat with advanced features."""
        print("\n🤖 Enhanced Knowledge Graph RAG System")
        print("=" * 60)
        print("Advanced Features:")
        print("• 'query <question>' - Standard query")
        print("• 'advanced <question>' - Advanced query with filtering")
        print("• 'visualize' - Show graph visualization")
        print("• 'analytics' - Show analytics dashboard")
        print("• 'stats' - Show graph statistics")
        print("• 'filter' - Set query filters")
        print("• 'entities <term>' - Find related entities")
        print("• 'path <entity1> <entity2>' - Find knowledge paths")
        print("• 'export' - Export graph data")
        print("• 'quit' - Exit")
        print("=" * 60)
        
        # Default query filter
        current_filter = QueryFilter()
        
        while True:
            try:
                user_input = input("\n💬 Your command: ").strip()
                
                if not user_input:
                    continue
                
                command_parts = user_input.split(' ', 1)
                command = command_parts[0].lower()
                args = command_parts[1] if len(command_parts) > 1 else ""
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif command == 'query' and args:
                    # Standard query
                    result = self.advanced_query(args, use_hybrid=False)
                    self._display_query_result(result)
                
                elif command == 'advanced' and args:
                    # Advanced query with current filter
                    result = self.advanced_query(args, query_filter=current_filter, use_hybrid=True)
                    self._display_query_result(result, show_advanced=True)
                
                elif command == 'visualize':
                    # Create and display graph visualization
                    print("🎨 Creating graph visualization...")
                    fig = self.visualize_graph(node_limit=50, show_labels=True)
                    if fig:
                        fig.show()
                    else:
                        print("❌ No graph available for visualization")
                
                elif command == 'analytics':
                    # Show analytics dashboard
                    print("📊 Creating analytics dashboard...")
                    dashboard = self.create_analytics_dashboard()
                    if dashboard:
                        dashboard.show()
                    else:
                        print("❌ No analytics available")
                
                elif command == 'stats':
                    # Show graph statistics
                    metrics = self.get_graph_analytics()
                    if metrics:
                        self._display_graph_stats(metrics)
                    else:
                        print("❌ No graph statistics available")
                
                elif command == 'filter':
                    # Set query filters
                    current_filter = self._configure_filters()
                
                elif command == 'entities' and args:
                    # Find related entities
                    if self.analytics:
                        related_entities = self.query_engine._find_related_entities(args)
                        print(f"\n🔍 Related entities for '{args}':")
                        for i, entity in enumerate(related_entities, 1):
                            print(f"  {i}. {entity}")
                    else:
                        print("❌ Analytics not available")
                
                elif command == 'path' and args:
                    # Find knowledge paths
                    parts = args.split()
                    if len(parts) >= 2:
                        entity1, entity2 = parts[0], parts[1]
                        if self.analytics:
                            path = self.analytics.find_shortest_path(entity1, entity2)
                            if path:
                                print(f"\n🛤️ Shortest path between '{entity1}' and '{entity2}':")
                                print(f"  {' -> '.join(path)}")
                            else:
                                print(f"❌ No path found between '{entity1}' and '{entity2}'")
                        else:
                            print("❌ Analytics not available")
                    else:
                        print("❌ Please provide two entities: path <entity1> <entity2>")
                
                elif command == 'export':
                    # Export graph data
                    if self.exporter:
                        export_info = self.exporter.export_graph_data()
                        print(f"✅ Graph data exported to {export_info['export_dir']}")
                        print(f"Files created: {len(export_info['files'])}")
                    else:
                        print("❌ Export functionality not available")
                
                elif command == 'help':
                    # Show help
                    self._show_help()
                
                else:
                    # Default to advanced query
                    if user_input:
                        result = self.advanced_query(user_input, query_filter=current_filter, use_hybrid=True)
                        self._display_query_result(result, show_advanced=True)
                    else:
                        print("❓ Unknown command. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Command processing error: {e}")
    
    def _display_query_result(self, result, show_advanced=False):
        """Display query result with formatting."""
        print(f"\n🤖 Answer: {result.answer}")
        
        if show_advanced and result.related_entities:
            print(f"\n🔗 Related entities: {', '.join(result.related_entities[:5])}")
        
        if show_advanced and result.knowledge_paths:
            print(f"\n🛤️ Knowledge paths:")
            for i, path in enumerate(result.knowledge_paths[:3], 1):
                print(f"  {i}. {' -> '.join(path)}")
        
        if result.sources:
            print(f"\n📚 Sources ({len(result.sources)}):")
            for i, source in enumerate(result.sources[:3], 1):
                source_type = source.get('source_type', 'unknown')
                score = source.get('score', 0.0)
                print(f"  {i}. [{source_type.upper()}] {source['text']} (Score: {score:.3f})")
    
    def _display_graph_stats(self, metrics):
        """Display graph statistics."""
        print(f"\n📊 Graph Statistics:")
        print(f"  Entities: {metrics.node_count}")
        print(f"  Relationships: {metrics.edge_count}")
        print(f"  Density: {metrics.density:.3f}")
        print(f"  Average Degree: {metrics.avg_degree:.2f}")
        print(f"  Clustering Coefficient: {metrics.clustering_coefficient:.3f}")
        print(f"  Communities: {len(metrics.communities)}")
        
        if metrics.centrality_scores.get('pagerank'):
            print(f"\n🏆 Top Entities (PageRank):")
            top_entities = sorted(
                metrics.centrality_scores['pagerank'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for i, (entity, score) in enumerate(top_entities, 1):
                print(f"  {i}. {entity}: {score:.3f}")
    
    def _configure_filters(self):
        """Interactive filter configuration."""
        print("\n🔧 Configure Query Filters:")
        print("Press Enter to skip any filter...")
        
        # Confidence threshold
        min_confidence = input("Minimum confidence (0.0-1.0): ").strip()
        min_confidence = float(min_confidence) if min_confidence else 0.0
        
        # Max hops
        max_hops = input("Maximum hops in graph (default 2): ").strip()
        max_hops = int(max_hops) if max_hops else 2
        
        # Document sources
        doc_sources = input("Document sources (comma-separated): ").strip()
        doc_sources = [s.strip() for s in doc_sources.split(',')] if doc_sources else None
        
        # Entity types
        entity_types = input("Entity types (comma-separated): ").strip()
        entity_types = [s.strip() for s in entity_types.split(',')] if entity_types else None
        
        filter_obj = QueryFilter(
            min_confidence=min_confidence,
            max_hops=max_hops,
            document_sources=doc_sources,
            entity_types=entity_types
        )
        
        print("✅ Filters configured!")
        return filter_obj
    
    def _show_help(self):
        """Show detailed help information."""
        print("\n📖 Help - Enhanced Knowledge Graph RAG System")
        print("=" * 60)
        print("BASIC COMMANDS:")
        print("  query <question>     - Ask a question using knowledge graph only")
        print("  advanced <question>  - Ask using hybrid KG + vector search")
        print("  <question>           - Same as advanced (default)")
        print()
        print("ANALYSIS COMMANDS:")
        print("  visualize           - Interactive graph visualization")
        print("  analytics           - Show analytics dashboard")
        print("  stats               - Display graph statistics")
        print("  entities <term>     - Find entities related to term")
        print("  path <e1> <e2>      - Find paths between two entities")
        print()
        print("CONFIGURATION:")
        print("  filter              - Configure query filters")
        print("  export              - Export graph data")
        print("  help                - Show this help")
        print("  quit                - Exit the system")
        print()
        print("QUERY EXAMPLES:")
        print("  • What are the main topics in machine learning?")
        print("  • How are neural networks related to deep learning?")
        print("  • entities artificial intelligence")
        print("  • path 'machine learning' 'neural networks'")
        print("=" * 60)


def main():
    """
    Main function to demonstrate the Enhanced Knowledge Graph RAG system.
    """
    print("🚀 Enhanced Knowledge Graph RAG System with Ollama")
    print("Features: Graph Visualization, Advanced Querying, Vector Database Integration")
    print("=" * 80)
    
    # Configuration
    config = KnowledgeGraphConfig(
        model_name='llama3.2',
        embedding_model='nomic-embed-text',
        vector_db_type='chroma',
        storage_dir='./enhanced_kg_storage',
        vector_db_path='./vector_db'
    )
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Initialize the enhanced system
        kg_rag = EnhancedKnowledgeGraphRAG(config)
        
        # Try to load existing indices
        if kg_rag.load_existing_indices():
            print("✅ Loaded existing knowledge graph and vector database")
        else:
            print("📁 No existing indices found. Please provide documents to build the knowledge graph.")
            data_dir = input("Documents directory path: ").strip()
            
            if not data_dir or not Path(data_dir).exists():
                print("❌ Invalid directory. Please create a 'data' folder with your documents.")
                print("Supported formats: PDF, TXT, MD, DOCX")
                return
            
            # Load and process documents
            print(f"📄 Loading documents from {data_dir}...")
            documents = kg_rag.load_documents(data_dir)
            
            if not documents:
                print("❌ No documents found. Please add documents to the directory.")
                return
            
            print(f"✅ Loaded {len(documents)} documents")
            
            # Build knowledge graph and vector index
            print("🔨 Building knowledge graph and vector database...")
            kg_rag.build_knowledge_graph(documents, build_vector_index=True)
            
            print("✅ Knowledge graph and vector database built successfully!")
        
        # Setup hybrid query engine
        print("🔍 Setting up hybrid query engine...")
        kg_rag.setup_hybrid_query_engine()
        
        # Show initial statistics
        metrics = kg_rag.get_graph_analytics()
        if metrics:
            print(f"\n📊 Graph Overview:")
            print(f"  • {metrics.node_count} entities")
            print(f"  • {metrics.edge_count} relationships")
            print(f"  • {len(metrics.communities)} communities detected")
            print(f"  • Graph density: {metrics.density:.3f}")
        
        # Start enhanced interactive chat
        kg_rag.interactive_chat_enhanced()
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("Please ensure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Required models are installed:")
        print("   - ollama pull llama3.2")
        print("   - ollama pull nomic-embed-text")
        print("3. Required Python packages are installed")


if __name__ == "__main__":
    main()
