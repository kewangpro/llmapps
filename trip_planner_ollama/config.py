from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

class APIConfig(BaseSettings):
    """Configuration management for travel APIs and application settings."""
    
    # Google Search API Configuration (optional - free tier available)
    google_search_api_key: Optional[str] = Field(None, env='GOOGLE_SEARCH_API_KEY')
    google_search_engine_id: Optional[str] = Field(None, env='GOOGLE_SEARCH_ENGINE_ID')
    
    # Cache Configuration
    cache_enabled: bool = Field(True, env='CACHE_ENABLED')
    cache_ttl_seconds: int = Field(3600, env='CACHE_TTL_SECONDS')  # 1 hour default
    redis_url: Optional[str] = Field(None, env='REDIS_URL')
    
    # Rate Limiting Configuration
    rate_limit_requests: int = Field(100, env='RATE_LIMIT_REQUESTS')
    rate_limit_period: int = Field(60, env='RATE_LIMIT_PERIOD')  # seconds
    
    # Fallback Configuration
    enable_fallback: bool = Field(True, env='ENABLE_FALLBACK')
    max_retries: int = Field(3, env='MAX_RETRIES')
    retry_delay: float = Field(1.0, env='RETRY_DELAY')  # seconds
    
    # Application Configuration
    debug: bool = Field(False, env='DEBUG')
    log_level: str = Field('INFO', env='LOG_LEVEL')
    
    # LLM Configuration
    ollama_model: str = Field('gemma3:latest', env='OLLAMA_MODEL')
    ollama_temperature: float = Field(0.3, env='OLLAMA_TEMPERATURE')
    ollama_max_iterations: int = Field(3, env='OLLAMA_MAX_ITERATIONS')
    agent_timeout: float = Field(120.0, env='AGENT_TIMEOUT')  # Increased timeout for complex agent tasks
    
    # Timeout Configuration
    api_timeout: float = Field(30.0, env='API_TIMEOUT')  # seconds
    connection_timeout: float = Field(10.0, env='CONNECTION_TIMEOUT')  # seconds
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate critical configuration settings."""
        warnings = []
        
        # Google Search API is optional - system works perfectly without it
        if not self.google_search_api_key:
            logger.info("Trip planner ready! Using intelligent travel data (no API keys needed).")
        else:
            logger.info("Google Search API configured. Enhanced with real-time search results.")
        
        if warnings:
            for warning in warnings:
                logger.warning(warning)
    
    @property
    def has_google_search_config(self) -> bool:
        """Check if Google Search API is properly configured."""
        return bool(self.google_search_api_key and self.google_search_engine_id)
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get configuration summary for debugging."""
        return {
            'google_search_configured': self.has_google_search_config,
            'cache_enabled': self.cache_enabled,
            'fallback_enabled': self.enable_fallback,
            'debug_mode': self.debug,
            'api_timeout': self.api_timeout,
            'rate_limit': f"{self.rate_limit_requests}/{self.rate_limit_period}s"
        }

# Global configuration instance
config = APIConfig()

def get_config() -> APIConfig:
    """Get the global configuration instance."""
    return config

def reload_config() -> APIConfig:
    """Reload configuration from environment."""
    global config
    config = APIConfig()
    return config