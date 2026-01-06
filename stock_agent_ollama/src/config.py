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
    REALTIME_DATA_TTL = 60  # 1 minute for real-time data
    BULK_DATA_TTL = 300     # 5 minutes for bulk data
    STOCK_INFO_TTL = 3600   # 1 hour for company info (fundamentals rarely change)
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

    # RL Trading settings
    RL_MODEL_DIR = MODEL_DIR / "rl"
    RL_DEFAULT_INITIAL_BALANCE = float(os.getenv("RL_INITIAL_BALANCE", "100000.0"))
    RL_TRANSACTION_COST_RATE = float(os.getenv("RL_TRANSACTION_COST", "0.0"))
    RL_SLIPPAGE_RATE = float(os.getenv("RL_SLIPPAGE", "0.0005"))
    RL_DEFAULT_TRAINING_TIMESTEPS = int(os.getenv("RL_TRAINING_TIMESTEPS", "300000"))
    RL_LOOKBACK_WINDOW = int(os.getenv("RL_LOOKBACK_WINDOW", "60"))
    RL_MAX_POSITION_PCT = float(os.getenv("RL_MAX_POSITION_PCT", "80.0"))
    RL_STOP_LOSS_PCT = float(os.getenv("RL_STOP_LOSS_PCT", "0.05"))
    RL_TRAILING_STOP_PCT = float(os.getenv("RL_TRAILING_STOP_PCT", "0.03"))
    RL_MAX_DRAWDOWN_PCT = float(os.getenv("RL_MAX_DRAWDOWN_PCT", "0.15"))

    # RL Enhancement Defaults
    RL_USE_ACTION_MASKING = True
    RL_USE_ENHANCED_REWARDS = True
    RL_USE_ADAPTIVE_SIZING = True
    RL_USE_IMPROVED_ACTIONS = True
    RL_ENABLE_DIAGNOSTICS = True

    # RL Agent default hyperparameters
    RL_PPO_LEARNING_RATE = float(os.getenv("RL_PPO_LR", "0.0003"))
    RL_A2C_LEARNING_RATE = float(os.getenv("RL_A2C_LR", "0.0007"))
    RL_GAMMA = float(os.getenv("RL_GAMMA", "0.99"))

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.CACHE_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.RL_MODEL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
