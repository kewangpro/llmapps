"""
LLM and embedding model setup for Ollama integration.
"""

import logging
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from models import KnowledgeGraphConfig

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM and embedding model setup."""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.llm = None
        self.embed_model = None
        self._setup_models()
    
    def _setup_models(self):
        """Setup Ollama models for LLM and embeddings."""
        try:
            self.llm = Ollama(
                model=self.config.model_name,
                base_url=self.config.ollama_url,
                temperature=0.1,
                request_timeout=120.0
            )
            
            self.embed_model = OllamaEmbedding(
                model_name=self.config.embedding_model,
                base_url=self.config.ollama_url
            )
            
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = self.config.chunk_size
            Settings.chunk_overlap = self.config.chunk_overlap
            
            logger.info(f"✅ Models initialized: {self.config.model_name}, {self.config.embedding_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test if Ollama connection is working."""
        try:
            # Simple test by calling the LLM
            response = self.llm.complete("Test connection")
            logger.info("✅ Ollama connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Ollama connection test failed: {e}")
            return False
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models."""
        try:
            import requests
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            logger.error(f"❌ Failed to get available models: {e}")
            return []
