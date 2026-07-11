"""
Knowledge graph building and management functionality.
"""

import logging
import networkx as nx
from typing import List, Tuple, Optional
from datetime import datetime

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, KnowledgeGraphIndex
from llama_index.core.node_parser import SentenceSplitter

from models import KnowledgeGraphConfig
from storage import StorageManager

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Handles knowledge graph construction and NetworkX graph building."""
    
    def __init__(self, config: KnowledgeGraphConfig, storage_manager: StorageManager):
        self.config = config
        self.storage_manager = storage_manager
        self.networkx_graph = None
    
    def build_knowledge_graph(
        self, 
        documents: List[Document], 
        include_embeddings: bool = True,
        build_vector_index: bool = True
    ) -> Tuple[KnowledgeGraphIndex, Optional[VectorStoreIndex]]:
        """Build both knowledge graph and vector indices."""
        try:
            logger.info("🔨 Building knowledge graph and vector index...")
            
            # Create storage context
            storage_context = self.storage_manager.create_storage_context(
                include_vector_store=build_vector_index
            )
            
            # Create node parser
            node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            # Build knowledge graph index
            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                max_triplets_per_chunk=self.config.max_triplets_per_chunk,
                node_parser=node_parser,
                include_embeddings=include_embeddings,
                show_progress=True
            )
            
            # Build vector index for hybrid search
            vector_index = None
            if build_vector_index:
                vector_index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    node_parser=node_parser,
                    show_progress=True
                )
            
            # Build NetworkX graph for analysis
            self._build_networkx_graph(kg_index)
            
            # Persist indices
            self.storage_manager.persist_indices(kg_index)
            
            logger.info("✅ Knowledge graph and vector index built successfully!")
            return kg_index, vector_index
            
        except Exception as e:
            logger.error(f"❌ Failed to build indices: {e}")
            raise
    
    def _build_networkx_graph(self, kg_index: KnowledgeGraphIndex):
        """Build NetworkX graph for advanced analysis and visualization."""
        try:
            self.networkx_graph = nx.Graph()
            
            graph_store = getattr(kg_index, 'graph_store', getattr(kg_index, '_graph_store', None))
            
            if graph_store:
                # Get all triplets from the graph store
                try:
                    # LlamaIndex SimpleGraphStore stores data differently
                    if hasattr(graph_store, 'graph_dict') and graph_store.graph_dict:
                        triplets = []
                        for subject, relations in graph_store.graph_dict.items():
                            for relation, objects in relations.items():
                                if isinstance(objects, list):
                                    for obj in objects:
                                        triplets.append((subject, relation, obj))
                                else:
                                    triplets.append((subject, relation, objects))
                    
                    elif hasattr(graph_store, 'get_triplets'):
                        triplets = graph_store.get_triplets()
                    
                    else:
                        # Try to extract from index directly
                        if hasattr(kg_index, 'get_networkx_graph'):
                            nx_graph = kg_index.get_networkx_graph()
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
    
    def get_networkx_graph(self) -> Optional[nx.Graph]:
        """Get the NetworkX graph representation."""
        return self.networkx_graph
    
    def rebuild_networkx_graph(self, kg_index: KnowledgeGraphIndex):
        """Rebuild NetworkX graph from knowledge graph index."""
        self._build_networkx_graph(kg_index)