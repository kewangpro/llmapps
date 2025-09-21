"""
Global LLM configuration for the multi-agent system
"""

import os
import logging
from llm_providers import create_llm, DEFAULT_MODELS
from typing import Optional

logger = logging.getLogger("LLMConfig")

class LLMConfig:
    """Singleton configuration for LLM provider"""

    _instance = None
    _provider = None
    _model = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMConfig, cls).__new__(cls)
        return cls._instance

    def configure(self, provider: str = "ollama", model: str = None):
        """Configure the LLM provider and model"""
        self._provider = provider.lower()

        # Set default model based on provider if not specified
        if model is None:
            if self._provider in DEFAULT_MODELS:
                model = list(DEFAULT_MODELS[self._provider].keys())[0]
            else:
                model = "gemma3:latest"  # fallback

        self._model = model
        self._llm = create_llm(self._provider, self._model)

        logger.info(f"🤖 LLM configured: {self._provider} / {self._model}")
        print(f"🤖 LLM configured: {self._provider} / {self._model}")

    def get_llm(self):
        """Get the configured LLM instance"""
        if self._llm is None:
            # Default configuration
            logger.info("🔧 No LLM configured, using default (ollama/gemma3:latest)")
            self.configure()
        return self._llm

    def get_provider(self) -> str:
        """Get current provider"""
        return self._provider or "ollama"

    def get_model(self) -> str:
        """Get current model"""
        return self._model or "gemma3:latest"

    def get_available_models(self):
        """Get all available models grouped by provider"""
        return DEFAULT_MODELS

# Global instance
llm_config = LLMConfig()