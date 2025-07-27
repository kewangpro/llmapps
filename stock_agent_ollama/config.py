from typing import Dict, Any
import logging
import os
import sys

# Configuration settings
CONFIG = {
    "ollama": {
        "model": "gemma3:latest",
        "base_url": "http://localhost:11434",
        "temperature": 0.1
    },
    "lstm": {
        "sequence_length": 120,
        "prediction_days": 30,
        "epochs": 75,
        "batch_size": 32
    },
    "stock_data": {
        "default_period": "2y",
        "supported_periods": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    },
    "streamlit": {
        "page_title": "Stock Analysis AI",
        "page_icon": "📈",
        "layout": "wide"
    }
}

def get_config() -> Dict[str, Any]:
    """Get application configuration"""
    return CONFIG

def update_config(key: str, value: Any) -> None:
    """Update configuration value"""
    keys = key.split('.')
    config = CONFIG
    for k in keys[:-1]:
        config = config[k]
    config[keys[-1]] = value

def setup_logging():
    """Configure logging for the entire application."""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('logs/stock_analysis.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific loggers to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    
    # Keep important loggers at INFO level
    logging.getLogger("stock_agent").setLevel(logging.INFO)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langchain.agents").setLevel(logging.INFO)

def get_logger(name):
    """Get a logger with the given name. Call setup_logging() first."""
    return logging.getLogger(name)