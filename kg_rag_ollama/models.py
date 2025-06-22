"""
Data models and classes for the Knowledge Graph RAG System.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import networkx as nx


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


@dataclass
class QueryResult:
    """Result from a knowledge graph query."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    related_entities: List[str]
    knowledge_paths: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class Triple:
    """Represents a knowledge triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0


@dataclass
class Entity:
    """Represents an entity with its type and mentions."""
    name: str
    entity_type: str
    mentions: List[str]
    confidence: float = 0.0


class KnowledgeGraphConfig:
    """Configuration class for the Knowledge Graph RAG system."""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        storage_dir: str = "./enhanced_kg_storage",
        vector_db_type: str = "chroma",
        vector_db_path: str = "./vector_db",
        max_triplets_per_chunk: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.storage_dir = Path(storage_dir)
        self.vector_db_type = vector_db_type
        self.vector_db_path = Path(vector_db_path)
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create storage directories
        self.storage_dir.mkdir(exist_ok=True)
        self.vector_db_path.mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_name': self.model_name,
            'embedding_model': self.embedding_model,
            'ollama_url': self.ollama_url,
            'storage_dir': str(self.storage_dir),
            'vector_db_type': self.vector_db_type,
            'vector_db_path': str(self.vector_db_path),
            'max_triplets_per_chunk': self.max_triplets_per_chunk,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'KnowledgeGraphConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)