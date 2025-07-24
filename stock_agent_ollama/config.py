from typing import Dict, Any

# Configuration settings
CONFIG = {
    "ollama": {
        "model": "gemma3:latest",
        "base_url": "http://localhost:11434",
        "temperature": 0.1
    },
    "lstm": {
        "sequence_length": 60,
        "prediction_days": 30,
        "epochs": 50,
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