"""
Query engine for knowledge graph and hybrid retrieval.
"""

import logging
import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime

from llama_index.core import KnowledgeGraphIndex, VectorStoreIndex

from models import QueryFilter, QueryResult, KnowledgeGraphConfig

logger = logging.getLogger(__name__)


class QueryEngine:
    """Handles query processing and hybrid retrieval."""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.kg_index = None
        self.vector_index = None
        self.kg_query_engine = None
        self.vector_query_engine = None
        self.hybrid_weight = 0.7
        self.networkx_graph = None
    
    def setup_query_engines(
        self,
        kg_index: KnowledgeGraphIndex,
        vector_index: Optional[VectorStoreIndex] = None,
        networkx_graph: Optional[nx.Graph] = None,
        kg_similarity_top_k: int = 3,
        vector_similarity_top_k: int = 5,
        hybrid_weight: float = 0.7
    ):
        """Setup hybrid query engine combining KG and vector retrieval."""
        try:
            self.kg_index = kg_index
            self.vector_index = vector_index
            self.networkx_graph = networkx_graph
            self.hybrid_weight = hybrid_weight
            
            if self.kg_index is None:
                raise ValueError("Knowledge graph index not initialized")
            
            # KG query engine
            self.kg_query_engine = self.kg_index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=kg_similarity_top_k,
                verbose=True
            )
            
            # Vector query engine (if available)
            if self.vector_index:
                self.vector_query_engine = self.vector_index.as_query_engine(
                    similarity_top_k=vector_similarity_top_k,
                    response_mode="tree_summarize"
                )
            
            logger.info("🔍 Query engines setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup query engine: {e}")
            raise
    
    def advanced_query(
        self, 
        question: str, 
        query_filter: Optional[QueryFilter] = None,
        use_hybrid: bool = True
    ) -> QueryResult:
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
            
            result = QueryResult(
                question=question,
                answer=combined_answer,
                sources=self._extract_source_info(results),
                related_entities=related_entities,
                knowledge_paths=knowledge_paths,
                metadata={
                    'query_type': 'hybrid' if use_hybrid else 'knowledge_graph',
                    'num_kg_sources': len([r for r in results if r['source'] == 'knowledge_graph']),
                    'num_vector_sources': len([r for r in results if r['source'] == 'vector_database']),
                    'model_used': self.config.model_name,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
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
                            try:
                                node_date = datetime.fromisoformat(node_date_str.replace('Z', '+00:00'))
                                if not (query_filter.date_range[0] <= node_date <= query_filter.date_range[1]):
                                    continue
                            except ValueError:
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
