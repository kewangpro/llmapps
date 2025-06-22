"""
Enhanced Property Graph Knowledge Graph RAG System
Implementation with graph visualization, advanced querying, and vector database integration.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# LlamaIndex imports
from llama_index.core import (
    Document, 
    SimpleDirectoryReader, 
    KnowledgeGraphIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    VectorStoreIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.query_engine import KnowledgeGraphQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever, VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore

# Vector store imports
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Ollama integration
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryFilter:
    """Advanced query filtering options."""
    entity_types: Optional[List[str]] = None
    relation_types: Optional[List[str]] = None
    min_confidence: float = 0.0
    date_range: Optional[Tuple[datetime, datetime]] = None
    document_sources: Optional[List[str]] = None
    max_hops: int = 2
    include_metadata: bool = True


@dataclass
class GraphMetrics:
    """Graph analysis metrics."""
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    clustering_coefficient: float
    centrality_scores: Dict[str, float]
    communities: List[List[str]]


class EnhancedKnowledgeGraphRAG:
    """
    Enhanced RAG system with visualization, advanced querying, and vector database integration.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        storage_dir: str = "./kg_storage",
        vector_db_type: str = "chroma",  # "chroma" or "qdrant"
        vector_db_path: str = "./vector_db",
        max_triplets_per_chunk: int = 10
    ):
        """
        Initialize the Enhanced Knowledge Graph RAG system.
        
        Args:
            vector_db_type: Type of vector database ("chroma" or "qdrant")
            vector_db_path: Path to vector database storage
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.storage_dir = Path(storage_dir)
        self.vector_db_type = vector_db_type
        self.vector_db_path = Path(vector_db_path)
        self.max_triplets_per_chunk = max_triplets_per_chunk
        
        # Create storage directories
        self.storage_dir.mkdir(exist_ok=True)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Initialize models
        self._setup_models()
        
        # Initialize vector database
        self._setup_vector_database()
        
        # Initialize components
        self.graph_store = None
        self.kg_index = None
        self.vector_index = None
        self.hybrid_query_engine = None
        self.networkx_graph = None
        
    def _setup_models(self):
        """Setup Ollama models for LLM and embeddings."""
        try:
            self.llm = Ollama(
                model=self.model_name,
                base_url=self.ollama_url,
                temperature=0.1,
                request_timeout=120.0
            )
            
            self.embed_model = OllamaEmbedding(
                model_name=self.embedding_model,
                base_url=self.ollama_url
            )
            
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            logger.info(f"✅ Models initialized: {self.model_name}, {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise
    
    def _setup_vector_database(self):
        """Setup vector database (Chroma or Qdrant)."""
        try:
            if self.vector_db_type.lower() == "chroma":
                # Initialize Chroma
                chroma_client = chromadb.PersistentClient(path=str(self.vector_db_path))
                chroma_collection = chroma_client.get_or_create_collection("knowledge_graph")
                self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                
            elif self.vector_db_type.lower() == "qdrant":
                # Initialize Qdrant
                qdrant_client = QdrantClient(path=str(self.vector_db_path))
                collection_name = "knowledge_graph"
                
                # Create collection if it doesn't exist
                try:
                    qdrant_client.get_collection(collection_name)
                except:
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # nomic-embed-text dimension
                    )
                
                self.vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=collection_name
                )
            else:
                raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")
            
            logger.info(f"✅ Vector database initialized: {self.vector_db_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            raise
    
    def load_documents(self, data_dir: str) -> List[Document]:
        """Load documents from directory with enhanced metadata."""
        try:
            reader = SimpleDirectoryReader(
                data_dir, 
                recursive=True,
                filename_as_id=True
            )
            documents = reader.load_data()
            
            # Add enhanced metadata
            for doc in documents:
                doc.metadata.update({
                    'source_type': Path(doc.metadata.get('file_name', '')).suffix,
                    'processed_date': datetime.now().isoformat(),
                    'chunk_count': len(doc.text) // Settings.chunk_size + 1
                })
            
            logger.info(f"📄 Loaded {len(documents)} documents from {data_dir}")
            return documents
        except Exception as e:
            logger.error(f"❌ Failed to load documents: {e}")
            return []
    
    def build_knowledge_graph(
        self, 
        documents: List[Document], 
        include_embeddings: bool = True,
        build_vector_index: bool = True
    ) -> Tuple[KnowledgeGraphIndex, Optional[VectorStoreIndex]]:
        """Build both knowledge graph and vector indices."""
        try:
            logger.info("🔨 Building knowledge graph and vector index...")
            
            # Initialize graph store
            self.graph_store = SimpleGraphStore()
            storage_context = StorageContext.from_defaults(
                graph_store=self.graph_store,
                vector_store=self.vector_store if build_vector_index else None
            )
            
            # Create node parser
            node_parser = SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap
            )
            
            # Build knowledge graph index
            self.kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                max_triplets_per_chunk=self.max_triplets_per_chunk,
                node_parser=node_parser,
                include_embeddings=include_embeddings,
                show_progress=True
            )
            
            # Build vector index for hybrid search
            if build_vector_index:
                self.vector_index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    node_parser=node_parser,
                    show_progress=True
                )
            
            # Build NetworkX graph for analysis
            self._build_networkx_graph()
            
            # Persist indices
            self.kg_index.storage_context.persist(persist_dir=str(self.storage_dir))
            
            logger.info("✅ Knowledge graph and vector index built successfully!")
            return self.kg_index, self.vector_index
            
        except Exception as e:
            logger.error(f"❌ Failed to build indices: {e}")
            raise

    def _build_networkx_graph(self):
        """Build NetworkX graph for advanced analysis and visualization."""
        try:
            self.networkx_graph = nx.Graph()
            
            if self.graph_store:
                # Get all triplets from the graph store
                try:
                    # LlamaIndex SimpleGraphStore stores data differently
                    if hasattr(self.graph_store, 'graph_dict') and self.graph_store.graph_dict:
                        triplets = []
                        for subject, relations in self.graph_store.graph_dict.items():
                            for relation, objects in relations.items():
                                if isinstance(objects, list):
                                    for obj in objects:
                                        triplets.append((subject, relation, obj))
                                else:
                                    triplets.append((subject, relation, objects))
                    
                    elif hasattr(self.graph_store, 'get_triplets'):
                        triplets = self.graph_store.get_triplets()
                    
                    else:
                        # Try to extract from index directly
                        if self.kg_index and hasattr(self.kg_index, 'get_networkx_graph'):
                            nx_graph = self.kg_index.get_networkx_graph()
                            if nx_graph:
                                self.networkx_graph = nx_graph
                                logger.info(f"📊 NetworkX graph loaded: {len(self.networkx_graph.nodes)} nodes, {len(self.networkx_graph.edges)} edges")
                                return
                        
                        logger.warning("⚠️ Cannot access graph data from graph store")
                        return
                    
                    if triplets:
                        for subject, predicate, obj in triplets:
                            # Clean and add nodes
                            subject_clean = str(subject).strip()
                            obj_clean = str(obj).strip()
                            predicate_clean = str(predicate).strip()
                            
                            if subject_clean and obj_clean:
                                self.networkx_graph.add_node(subject_clean, type='entity')
                                self.networkx_graph.add_node(obj_clean, type='entity')
                                
                                # Add edge with relationship type
                                self.networkx_graph.add_edge(
                                    subject_clean, obj_clean, 
                                    relation=predicate_clean,
                                    weight=1.0
                                )
                        
                        logger.info(f"📊 NetworkX graph built: {len(self.networkx_graph.nodes)} nodes, {len(self.networkx_graph.edges)} edges")
                    else:
                        logger.warning("⚠️ No triplets found in graph store")
                        
                except Exception as e:
                    logger.error(f"❌ Error extracting triplets: {e}")
                    
            else:
                logger.warning("⚠️ Graph store not initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to build NetworkX graph: {e}")
            # Create empty graph as fallback
            self.networkx_graph = nx.Graph()

    def load_existing_indices(self) -> bool:
        """Load existing knowledge graph and vector indices."""
        try:
            if not (self.storage_dir / "index_store.json").exists():
                logger.info("📂 No existing indices found")
                return False
            
            # Load storage context with vector store
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_dir),
                vector_store=self.vector_store
            )
            
            # Check what indices are available
            index_store = storage_context.index_store
            
            # Handle different index_structs formats
            index_ids = []
            try:
                # index_structs is a method that returns the actual structures
                if hasattr(index_store, 'index_structs'):
                    structs = index_store.index_structs()
                    if isinstance(structs, dict):
                        index_ids = list(structs.keys())
                    elif isinstance(structs, list):
                        # Extract index_id from each index struct object
                        for struct in structs:
                            if hasattr(struct, 'index_id'):
                                index_ids.append(struct.index_id)
                            else:
                                # Fallback to position-based ID
                                index_ids.append(f"index_{len(index_ids)}")
                    else:
                        logger.warning("⚠️ Unexpected index_structs format, using fallback")
                        index_ids = ['default']
                else:
                    logger.warning("⚠️ No index_structs method found, using fallback")
                    index_ids = ['default']
            except Exception as e:
                logger.warning(f"⚠️ Error accessing index_structs: {e}, using fallback")
                index_ids = ['default']
            
            logger.info(f"Found {len(index_ids)} indices: {index_ids}")
            
            # Load knowledge graph index
            kg_loaded = False
            for index_id in index_ids:
                try:
                    # Always specify index_id when loading to avoid ambiguity
                    if index_id != 'default':
                        index = load_index_from_storage(storage_context, index_id=index_id)
                    else:
                        # For fallback case, try to get the first index from the store
                        if len(index_ids) == 1:
                            # Only one index, safe to load without specifying ID
                            index = load_index_from_storage(storage_context)
                        else:
                            # Skip default when multiple indices exist
                            continue
                    
                    # Check if this is a knowledge graph index
                    if (hasattr(index, 'graph_store') or 
                        'knowledge' in str(index_id).lower() or 
                        'kg' in str(index_id).lower() or
                        hasattr(index, '_graph_store')):
                        
                        self.kg_index = index
                        self.graph_store = getattr(index, 'graph_store', getattr(index, '_graph_store', None))
                        kg_loaded = True
                        logger.info(f"✅ Loaded knowledge graph index: {index_id}")
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load index {index_id}: {e}")
                    continue
            
            # If no specific KG index found, try loading the first available index with explicit ID
            if not kg_loaded and index_ids:
                for index_id in index_ids:
                    if index_id == 'default':
                        continue  # Skip default when multiple indices exist
                    try:
                        self.kg_index = load_index_from_storage(storage_context, index_id=index_id)
                        self.graph_store = getattr(self.kg_index, 'graph_store', getattr(self.kg_index, '_graph_store', None))
                        kg_loaded = True
                        logger.info(f"✅ Loaded index as knowledge graph: {index_id}")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load index {index_id}: {e}")
                        continue
                
                # Final fallback for single index case
                if not kg_loaded and len(index_ids) == 1 and index_ids[0] == 'default':
                    try:
                        self.kg_index = load_index_from_storage(storage_context)
                        self.graph_store = getattr(self.kg_index, 'graph_store', getattr(self.kg_index, '_graph_store', None))
                        kg_loaded = True
                        logger.info("✅ Loaded single default index as knowledge graph")
                    except Exception as e:
                        logger.error(f"❌ Failed to load default index: {e}")
                        return False
                
                if not kg_loaded:
                    logger.error("❌ Failed to load any knowledge graph index")
                    return False
            
            # Try to load vector index (if different from KG index)
            if len(index_ids) > 1:
                for index_id in index_ids:
                    try:
                        # Skip the already loaded KG index and default entries
                        if (index_id == getattr(self.kg_index, 'index_id', None) or 
                            index_id == 'default'):
                            continue
                        
                        # Always specify index_id to avoid ambiguity
                        vector_index = load_index_from_storage(storage_context, index_id=index_id)
                        
                        # Check if this looks like a vector index
                        if (hasattr(vector_index, 'vector_store') or 
                            'vector' in str(index_id).lower() or
                            not hasattr(vector_index, 'graph_store')):  # If it's not a KG index, likely vector
                            self.vector_index = vector_index
                            logger.info(f"✅ Loaded vector index: {index_id}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load vector index {index_id}: {e}")
                        continue
            
            if not self.vector_index:
                logger.warning("⚠️ Vector index not found, will create if needed")
            
            # Build NetworkX graph from the loaded data
            self._build_networkx_graph()
            
            logger.info("✅ Existing indices loaded successfully!")
            return kg_loaded
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing indices: {e}")
            return False
 
    def setup_hybrid_query_engine(
        self, 
        kg_similarity_top_k: int = 3,
        vector_similarity_top_k: int = 5,
        hybrid_weight: float = 0.7  # Weight for KG vs vector retrieval
    ):
        """Setup hybrid query engine combining KG and vector retrieval."""
        try:
            if self.kg_index is None:
                raise ValueError("Knowledge graph index not initialized")
            
            # KG query engine
            kg_query_engine = self.kg_index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=kg_similarity_top_k,
                verbose=True
            )
            
            # Vector query engine (if available)
            vector_query_engine = None
            if self.vector_index:
                vector_query_engine = self.vector_index.as_query_engine(
                    similarity_top_k=vector_similarity_top_k,
                    response_mode="tree_summarize"
                )
            
            # Store both engines
            self.kg_query_engine = kg_query_engine
            self.vector_query_engine = vector_query_engine
            self.hybrid_weight = hybrid_weight
            
            logger.info("🔍 Hybrid query engine setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup query engine: {e}")
            raise
    
    def advanced_query(
        self, 
        question: str, 
        query_filter: Optional[QueryFilter] = None,
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Advanced query with filtering and hybrid retrieval.
        
        Args:
            question: User's question
            query_filter: Advanced filtering options
            use_hybrid: Whether to use hybrid KG + vector retrieval
        """
        try:
            logger.info(f"❓ Processing advanced query: {question}")
            
            results = []
            
            # Knowledge Graph Query
            if self.kg_query_engine:
                kg_response = self.kg_query_engine.query(question)
                results.append({
                    'source': 'knowledge_graph',
                    'response': str(kg_response),
                    'source_nodes': getattr(kg_response, 'source_nodes', []),
                    'weight': self.hybrid_weight
                })
            
            # Vector Database Query
            if use_hybrid and self.vector_query_engine:
                vector_response = self.vector_query_engine.query(question)
                results.append({
                    'source': 'vector_database',
                    'response': str(vector_response),
                    'source_nodes': getattr(vector_response, 'source_nodes', []),
                    'weight': 1.0 - self.hybrid_weight
                })
            
            # Apply filters if provided
            if query_filter:
                results = self._apply_query_filters(results, query_filter)
            
            # Combine results
            combined_answer = self._combine_responses(results)
            
            # Get related entities and paths
            related_entities = self._find_related_entities(question)
            knowledge_paths = self._find_knowledge_paths(question, max_hops=query_filter.max_hops if query_filter else 2)
            
            result = {
                'question': question,
                'answer': combined_answer,
                'sources': self._extract_source_info(results),
                'related_entities': related_entities,
                'knowledge_paths': knowledge_paths,
                'metadata': {
                    'query_type': 'hybrid' if use_hybrid else 'knowledge_graph',
                    'num_kg_sources': len([r for r in results if r['source'] == 'knowledge_graph']),
                    'num_vector_sources': len([r for r in results if r['source'] == 'vector_database']),
                    'model_used': self.model_name,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info("✅ Advanced query processed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process advanced query: {e}")
            raise
    
    def _apply_query_filters(self, results: List[Dict], query_filter: QueryFilter) -> List[Dict]:
        """Apply advanced filters to query results."""
        filtered_results = []
        
        for result in results:
            # Filter by confidence score
            source_nodes = result.get('source_nodes', [])
            filtered_nodes = []
            
            for node in source_nodes:
                score = getattr(node, 'score', 1.0)
                if score >= query_filter.min_confidence:
                    # Additional metadata filtering
                    node_metadata = getattr(node, 'metadata', {})
                    
                    # Filter by document sources
                    if query_filter.document_sources:
                        source_file = node_metadata.get('file_name', '')
                        if not any(src in source_file for src in query_filter.document_sources):
                            continue
                    
                    # Filter by date range
                    if query_filter.date_range:
                        node_date_str = node_metadata.get('processed_date')
                        if node_date_str:
                            node_date = datetime.fromisoformat(node_date_str.replace('Z', '+00:00'))
                            if not (query_filter.date_range[0] <= node_date <= query_filter.date_range[1]):
                                continue
                    
                    filtered_nodes.append(node)
            
            if filtered_nodes:
                result['source_nodes'] = filtered_nodes
                filtered_results.append(result)
        
        return filtered_results
    
    def _combine_responses(self, results: List[Dict]) -> str:
        """Combine multiple query responses using weighted synthesis."""
        if not results:
            return "No relevant information found."
        
        if len(results) == 1:
            return results[0]['response']
        
        # Weight responses and combine
        combined_text = []
        for result in results:
            weight = result.get('weight', 1.0)
            source = result['source']
            response = result['response']
            
            combined_text.append(f"From {source} (weight: {weight:.2f}):\n{response}")
        
        return "\n\n".join(combined_text)
    
    def _extract_source_info(self, results: List[Dict]) -> List[Dict]:
        """Extract and format source information."""
        sources = []
        
        for result in results:
            source_nodes = result.get('source_nodes', [])
            for node in source_nodes:
                sources.append({
                    'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'score': getattr(node, 'score', 0.0),
                    'metadata': getattr(node, 'metadata', {}),
                    'source_type': result['source']
                })
        
        return sources
    
    def _find_related_entities(self, question: str, top_k: int = 10) -> List[str]:
        """Find entities related to the question using graph analysis."""
        if not self.networkx_graph:
            return []
        
        try:
            # Simple keyword matching for now - could be enhanced with NLP
            question_words = set(question.lower().split())
            related_nodes = []
            
            for node in self.networkx_graph.nodes():
                node_words = set(node.lower().split())
                if question_words.intersection(node_words):
                    # Get neighbors
                    neighbors = list(self.networkx_graph.neighbors(node))
                    related_nodes.extend([node] + neighbors[:3])
            
            return list(set(related_nodes))[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Failed to find related entities: {e}")
            return []
    
    def _find_knowledge_paths(self, question: str, max_hops: int = 2) -> List[List[str]]:
        """Find knowledge paths between entities mentioned in the question."""
        if not self.networkx_graph:
            return []
        
        try:
            # Extract potential entities from question
            question_entities = self._extract_entities_from_text(question)
            
            paths = []
            for i, entity1 in enumerate(question_entities):
                for entity2 in question_entities[i+1:]:
                    if entity1 in self.networkx_graph and entity2 in self.networkx_graph:
                        try:
                            shortest_path = nx.shortest_path(
                                self.networkx_graph, 
                                entity1, 
                                entity2,
                                weight=None
                            )
                            if len(shortest_path) <= max_hops + 1:
                                paths.append(shortest_path)
                        except nx.NetworkXNoPath:
                            continue
            
            return paths[:5]  # Return top 5 paths
            
        except Exception as e:
            logger.error(f"❌ Failed to find knowledge paths: {e}")
            return []
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Simple entity extraction - could be enhanced with NER."""
        # For now, use simple capitalized word extraction
        words = text.split()
        entities = []
        
        for word in words:
            cleaned = word.strip('.,!?;:"()[]{}')
            if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
                entities.append(cleaned)
        
        # Filter entities that exist in the graph
        if self.networkx_graph:
            entities = [e for e in entities if e in self.networkx_graph.nodes()]
        
        return entities
    
    def get_graph_analytics(self) -> GraphMetrics:
        """Get comprehensive graph analytics."""
        try:
            if not self.networkx_graph:
                raise ValueError("NetworkX graph not available")
            
            G = self.networkx_graph
            
            # Basic metrics
            node_count = len(G.nodes())
            edge_count = len(G.edges())
            density = nx.density(G)
            
            # Degree statistics
            degrees = dict(G.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            
            # Clustering coefficient
            clustering_coeff = nx.average_clustering(G)
            
            # Centrality measures
            centrality_scores = {}
            if node_count > 0 and edge_count > 0:
                centrality_scores['betweenness'] = nx.betweenness_centrality(G)
                centrality_scores['closeness'] = nx.closeness_centrality(G)
                centrality_scores['pagerank'] = nx.pagerank(G)
            
            # Community detection
            communities = []
            if edge_count > 0:
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = list(nx_comm.greedy_modularity_communities(G))
                    communities = [list(community) for community in communities]
                except:
                    logger.warning("⚠️ Community detection failed")
            
            return GraphMetrics(
                node_count=node_count,
                edge_count=edge_count,
                density=density,
                avg_degree=avg_degree,
                clustering_coefficient=clustering_coeff,
                centrality_scores=centrality_scores,
                communities=communities
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get graph analytics: {e}")
            return GraphMetrics(0, 0, 0, 0, 0, {}, [])
    
    def visualize_graph(
        self, 
        layout: str = "spring",
        node_limit: int = 100,
        show_labels: bool = True,
        highlight_entities: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive graph visualization using Plotly.
        
        Args:
            layout: Graph layout algorithm ("spring", "circular", "random")
            node_limit: Maximum number of nodes to display
            show_labels: Whether to show node labels
            highlight_entities: List of entities to highlight
            save_path: Path to save the visualization
        """
        try:
            if not self.networkx_graph or len(self.networkx_graph.nodes()) == 0:
                # Try to rebuild the graph
                logger.info("🔄 Attempting to rebuild NetworkX graph...")
                self._build_networkx_graph()
                
                if not self.networkx_graph or len(self.networkx_graph.nodes()) == 0:
                    raise ValueError("No graph data available for visualization. Please build the knowledge graph first.")

            G = self.networkx_graph
            
            # Limit nodes for performance
            if len(G.nodes()) > node_limit:
                # Get most central nodes
                centrality = nx.degree_centrality(G)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:node_limit]
                top_node_names = [node[0] for node in top_nodes]
                G = G.subgraph(top_node_names)
            
            # Generate layout
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.random_layout(G)
            
            # Extract edges
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Get edge information
                edge_data = G.get_edge_data(edge[0], edge[1])
                relation = edge_data.get('relation', 'connected') if edge_data else 'connected'
                edge_info.append(f"{edge[0]} -> {relation} -> {edge[1]}")
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Relationships'
            )
            
            # Extract nodes
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            node_hover_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node information
                adjacencies = list(G.neighbors(node))
                degree = len(adjacencies)
                
                # Create hover text
                hover_info = f'<b>{node}</b><br>'
                hover_info += f'Connections: {degree}<br>'
                if adjacencies:
                    hover_info += f'Connected to: {", ".join(adjacencies[:3])}'
                    if len(adjacencies) > 3:
                        hover_info += f' and {len(adjacencies) - 3} more...'
                
                node_hover_text.append(hover_info)
                
                # Set node text for labels
                node_text.append(node if show_labels else '')
                
                # Color and size nodes based on degree and highlights
                if highlight_entities and node in highlight_entities:
                    node_colors.append('#ff4444')  # Red for highlighted
                    node_sizes.append(max(15, min(30, degree * 2)))
                else:
                    # Color based on degree (more connections = darker)
                    intensity = min(degree / 10.0, 1.0)  # Normalize degree
                    node_colors.append(f'rgba(100, 149, 237, {0.3 + 0.7 * intensity})')
                    node_sizes.append(max(8, min(20, degree + 5)))
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if show_labels else 'markers',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                hovertext=node_hover_text,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                name='Entities'
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f'Knowledge Graph Visualization ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                        font=dict(size=16)
                    ),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=60),
                    annotations=[
                        dict(
                            text="Interactive Knowledge Graph - Hover over nodes for details",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="grey", size=10)
                        )
                    ],
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        scaleanchor="y",
                        scaleratio=1
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12)
                )
            )
            
            # Add some styling for better appearance
            fig.update_layout(
                width=1000,
                height=800,
                dragmode='pan'
            )
            
            # Save if requested
            if save_path:
                fig.write_html(save_path)
                logger.info(f"💾 Graph visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create graph visualization: {e}")
            raise
    
    def create_analytics_dashboard(self) -> go.Figure:
        """Create comprehensive analytics dashboard."""
        try:
            metrics = self.get_graph_analytics()
            
            # Create subplots with mixed subplot types
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Graph Overview', 
                    'Node Degree Distribution',
                    'Top Entities by Centrality',
                    'Community Sizes'
                ),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "domain"}]]  # Changed last one to domain for pie chart
            )
            
            # 1. Graph Overview (Key Metrics as Bar Chart)
            overview_metrics = ['Entities', 'Relationships', 'Communities', 'Avg Degree']
            overview_values = [
                metrics.node_count, 
                metrics.edge_count, 
                len(metrics.communities), 
                round(metrics.avg_degree, 1)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=overview_metrics,
                    y=overview_values,
                    name="Graph Metrics",
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                ),
                row=1, col=1
            )
            
            # 2. Degree distribution
            if self.networkx_graph and len(self.networkx_graph.nodes()) > 0:
                degrees = [d for n, d in self.networkx_graph.degree()]
                fig.add_trace(
                    go.Histogram(
                        x=degrees, 
                        name="Degree Distribution",
                        nbinsx=min(20, max(degrees) + 1) if degrees else 10,
                        marker_color='#ff7f0e'
                    ),
                    row=1, col=2
                )
            else:
                # Empty placeholder
                fig.add_trace(
                    go.Bar(x=['No Data'], y=[0], name="No Data Available"),
                    row=1, col=2
                )
            
            # 3. Top entities by centrality
            if metrics.centrality_scores.get('pagerank') and len(metrics.centrality_scores['pagerank']) > 0:
                top_entities = sorted(
                    metrics.centrality_scores['pagerank'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                if top_entities:
                    entities, scores = zip(*top_entities)
                    
                    # Truncate long entity names for display
                    display_entities = [
                        entity[:20] + '...' if len(entity) > 20 else entity 
                        for entity in entities
                    ]
                    
                    fig.add_trace(
                        go.Bar(
                            x=display_entities, 
                            y=list(scores), 
                            name="PageRank Centrality",
                            marker_color='#2ca02c',
                            hovertext=[f"{entity}: {score:.4f}" for entity, score in zip(entities, scores)],
                            hoverinfo='text'
                        ),
                        row=2, col=1
                    )
                else:
                    fig.add_trace(
                        go.Bar(x=['No Data'], y=[0], name="No Centrality Data"),
                        row=2, col=1
                    )
            else:
                fig.add_trace(
                    go.Bar(x=['No Data'], y=[0], name="No Centrality Data"),
                    row=2, col=1
                )
            
            # 4. Community sizes - Now using pie chart in domain subplot
            if metrics.communities and len(metrics.communities) > 0:
                community_sizes = [len(community) for community in metrics.communities]
                community_labels = [f"Community {i+1}" for i in range(len(community_sizes))]
                
                # Only show if we have meaningful communities
                if max(community_sizes) > 1:
                    fig.add_trace(
                        go.Pie(
                            labels=community_labels, 
                            values=community_sizes, 
                            name="Communities",
                            hoverinfo='label+value+percent'
                        ),
                        row=2, col=2
                    )
                else:
                    # Use bar chart instead for single nodes
                    fig.add_trace(
                        go.Bar(x=['Single Nodes'], y=[len(community_sizes)], name="No Communities"),
                        row=2, col=2
                    )
            else:
                # Use bar chart for no community data
                fig.add_trace(
                    go.Bar(x=['No Communities'], y=[0], name="No Community Data"),
                    row=2, col=2
                )
            
            # Update layout with proper title formatting
            fig.update_layout(
                title=dict(
                    text="Knowledge Graph Analytics Dashboard",
                    font=dict(size=20)
                ),
                showlegend=False,
                height=800,
                width=1200,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Update x-axis labels for readability
            fig.update_xaxes(tickangle=45, row=2, col=1)  # Rotate entity names
            
            # Add some spacing between subplots
            fig.update_layout(
                margin=dict(t=80, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create analytics dashboard: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating dashboard: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Analytics Dashboard - Error",
                width=800, height=400
            )
            return fig
    
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
                    fig.show()
                
                elif command == 'analytics':
                    # Show analytics dashboard
                    print("📊 Creating analytics dashboard...")
                    dashboard = self.create_analytics_dashboard()
                    dashboard.show()
                
                elif command == 'stats':
                    # Show graph statistics
                    metrics = self.get_graph_analytics()
                    self._display_graph_stats(metrics)
                
                elif command == 'filter':
                    # Set query filters
                    current_filter = self._configure_filters()
                
                elif command == 'entities' and args:
                    # Find related entities
                    entities = self._find_related_entities(args)
                    print(f"\n🔍 Related entities for '{args}':")
                    for i, entity in enumerate(entities, 1):
                        print(f"  {i}. {entity}")
                
                elif command == 'path' and args:
                    # Find knowledge paths
                    parts = args.split()
                    if len(parts) >= 2:
                        entity1, entity2 = parts[0], parts[1]
                        paths = self._find_knowledge_paths(f"{entity1} {entity2}")
                        print(f"\n🛤️ Knowledge paths between '{entity1}' and '{entity2}':")
                        for i, path in enumerate(paths, 1):
                            print(f"  {i}. {' -> '.join(path)}")
                    else:
                        print("❌ Please provide two entities: path <entity1> <entity2>")
                
                elif command == 'export':
                    # Export graph data
                    self._export_graph_data()
                
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
    
    def _display_query_result(self, result: Dict[str, Any], show_advanced: bool = False):
        """Display query result with formatting."""
        print(f"\n🤖 Answer: {result['answer']}")
        
        if show_advanced and result.get('related_entities'):
            print(f"\n🔗 Related entities: {', '.join(result['related_entities'][:5])}")
        
        if show_advanced and result.get('knowledge_paths'):
            print(f"\n🛤️ Knowledge paths:")
            for i, path in enumerate(result['knowledge_paths'][:3], 1):
                print(f"  {i}. {' -> '.join(path)}")
        
        if result.get('sources'):
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:3], 1):
                source_type = source.get('source_type', 'unknown')
                score = source.get('score', 0.0)
                print(f"  {i}. [{source_type.upper()}] {source['text']} (Score: {score:.3f})")
    
    def _display_graph_stats(self, metrics: GraphMetrics):
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
    
    def _configure_filters(self) -> QueryFilter:
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
    
    def _export_graph_data(self):
        """Export graph data to various formats."""
        try:
            export_dir = Path("./exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export NetworkX graph
            if self.networkx_graph:
                # JSON format
                graph_data = nx.node_link_data(self.networkx_graph)
                with open(export_dir / f"graph_{timestamp}.json", 'w') as f:
                    json.dump(graph_data, f, indent=2)
                
                # CSV format for edges
                edges_df = pd.DataFrame([
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relation': self.networkx_graph.get_edge_data(edge[0], edge[1]).get('relation', 'connected')
                    }
                    for edge in self.networkx_graph.edges()
                ])
                edges_df.to_csv(export_dir / f"edges_{timestamp}.csv", index=False)
                
                # CSV format for nodes
                nodes_df = pd.DataFrame([
                    {
                        'node': node,
                        'degree': self.networkx_graph.degree(node),
                        'type': self.networkx_graph.nodes[node].get('type', 'entity')
                    }
                    for node in self.networkx_graph.nodes()
                ])
                nodes_df.to_csv(export_dir / f"nodes_{timestamp}.csv", index=False)
            
            # Export analytics
            metrics = self.get_graph_analytics()
            analytics_data = {
                'timestamp': timestamp,
                'node_count': metrics.node_count,
                'edge_count': metrics.edge_count,
                'density': metrics.density,
                'avg_degree': metrics.avg_degree,
                'clustering_coefficient': metrics.clustering_coefficient,
                'communities': metrics.communities
            }
            
            with open(export_dir / f"analytics_{timestamp}.json", 'w') as f:
                json.dump(analytics_data, f, indent=2)
            
            print(f"✅ Graph data exported to {export_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export graph data: {e}")
    
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
    config = {
        'model_name': 'llama3.2',
        'embedding_model': 'nomic-embed-text',
        'vector_db_type': 'chroma',  # or 'qdrant'
        'storage_dir': './enhanced_kg_storage',
        'vector_db_path': './vector_db'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Initialize the enhanced system
        kg_rag = EnhancedKnowledgeGraphRAG(**config)
        
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