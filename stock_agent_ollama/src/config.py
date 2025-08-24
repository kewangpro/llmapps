import os
from pathlib import Path
from typing import Optional

class Config:
    """Application configuration with environment variable support"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = DATA_DIR / "cache"
    MODEL_DIR = DATA_DIR / "models"
    LOG_DIR = DATA_DIR / "logs"
    
    # Panel configuration
    PANEL_PORT = int(os.getenv("PANEL_PORT", "5006"))
    PANEL_HOST = os.getenv("PANEL_HOST", "localhost")
    PANEL_ALLOW_WEBSOCKET_ORIGIN = [f"{PANEL_HOST}:{PANEL_PORT}"]
    
    # Cache settings
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    STOCK_DATA_TTL = 900  # 15 minutes for real-time data
    HISTORICAL_DATA_TTL = 86400  # 1 day for historical data
    
    # LSTM model settings
    LSTM_SEQUENCE_LENGTH = 60
    LSTM_ENSEMBLE_SIZE = 3
    PREDICTION_DAYS = 30
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Ollama configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
    OLLAMA_FALLBACK_TO_REGEX = os.getenv("OLLAMA_FALLBACK_TO_REGEX", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.CACHE_DIR, cls.MODEL_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
