"""
Storage management for knowledge graphs and vector databases.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

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
from llama_index.core.graph_stores import SimpleGraphStore

# Vector store imports
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from datetime import datetime
from models import KnowledgeGraphConfig

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage operations for knowledge graphs and vector databases."""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.vector_store = None
        self.graph_store = None
        self._setup_vector_database()
    
    def _setup_vector_database(self):
        """Setup vector database (Chroma or Qdrant)."""
        try:
            if self.config.vector_db_type.lower() == "chroma":
                # Initialize Chroma
                chroma_client = chromadb.PersistentClient(path=str(self.config.vector_db_path))
                chroma_collection = chroma_client.get_or_create_collection("knowledge_graph")
                self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                
            elif self.config.vector_db_type.lower() == "qdrant":
                # Initialize Qdrant
                qdrant_client = QdrantClient(path=str(self.config.vector_db_path))
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
                raise ValueError(f"Unsupported vector database type: {self.config.vector_db_type}")
            
            logger.info(f"✅ Vector database initialized: {self.config.vector_db_type}")
            
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
                    'processed_date': f"{datetime.now().isoformat()}",
                    'chunk_count': len(doc.text) // Settings.chunk_size + 1
                })
            
            logger.info(f"📄 Loaded {len(documents)} documents from {data_dir}")
            return documents
        except Exception as e:
            logger.error(f"❌ Failed to load documents: {e}")
            return []
    
    def create_storage_context(self, include_vector_store: bool = True) -> StorageContext:
        """Create storage context with graph and vector stores."""
        self.graph_store = SimpleGraphStore()
        return StorageContext.from_defaults(
            graph_store=self.graph_store,
            vector_store=self.vector_store if include_vector_store else None
        )
    
    def persist_indices(self, kg_index: KnowledgeGraphIndex):
        """Persist knowledge graph index to storage."""
        try:
            kg_index.storage_context.persist(persist_dir=str(self.config.storage_dir))
            logger.info(f"✅ Indices persisted to {self.config.storage_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to persist indices: {e}")
            raise
    
    def load_existing_indices(self) -> tuple[Optional[KnowledgeGraphIndex], Optional[VectorStoreIndex]]:
        """Load existing knowledge graph and vector indices."""
        try:
            if not (self.config.storage_dir / "index_store.json").exists():
                logger.info("📂 No existing indices found")
                return None, None
            
            # Load storage context with vector store
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.config.storage_dir),
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
            
            kg_index = None
            vector_index = None
            
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
                        
                        kg_index = index
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
                        kg_index = load_index_from_storage(storage_context, index_id=index_id)
                        self.graph_store = getattr(kg_index, 'graph_store', getattr(kg_index, '_graph_store', None))
                        kg_loaded = True
                        logger.info(f"✅ Loaded index as knowledge graph: {index_id}")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load index {index_id}: {e}")
                        continue
                
                # Final fallback for single index case
                if not kg_loaded and len(index_ids) == 1 and index_ids[0] == 'default':
                    try:
                        kg_index = load_index_from_storage(storage_context)
                        self.graph_store = getattr(kg_index, 'graph_store', getattr(kg_index, '_graph_store', None))
                        kg_loaded = True
                        logger.info("✅ Loaded single default index as knowledge graph")
                    except Exception as e:
                        logger.error(f"❌ Failed to load default index: {e}")
                        return None, None
                
                if not kg_loaded:
                    logger.error("❌ Failed to load any knowledge graph index")
                    return None, None
            
            # Try to load vector index (if different from KG index)
            if len(index_ids) > 1:
                for index_id in index_ids:
                    try:
                        # Skip the already loaded KG index and default entries
                        if (index_id == getattr(kg_index, 'index_id', None) or 
                            index_id == 'default'):
                            continue
                        
                        # Always specify index_id to avoid ambiguity
                        vector_index_candidate = load_index_from_storage(storage_context, index_id=index_id)
                        
                        # Check if this looks like a vector index
                        if (hasattr(vector_index_candidate, 'vector_store') or 
                            'vector' in str(index_id).lower() or
                            not hasattr(vector_index_candidate, 'graph_store')):  # If it's not a KG index, likely vector
                            vector_index = vector_index_candidate
                            logger.info(f"✅ Loaded vector index: {index_id}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load vector index {index_id}: {e}")
                        continue
            
            if not vector_index:
                logger.warning("⚠️ Vector index not found, will create if needed")
            
            logger.info("✅ Existing indices loaded successfully!")
            return kg_index, vector_index
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing indices: {e}")
            return None, None
