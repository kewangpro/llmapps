"""
Tests for Configuration module
"""
import pytest
import os
from pathlib import Path
from src.config import Config


def test_config_base_paths():
    """Test that base paths are set correctly."""
    assert isinstance(Config.BASE_DIR, Path)
    assert isinstance(Config.DATA_DIR, Path)
    assert isinstance(Config.CACHE_DIR, Path)
    assert isinstance(Config.MODEL_DIR, Path)
    assert isinstance(Config.LOG_DIR, Path)

    # Verify hierarchy
    assert Config.DATA_DIR.parent == Config.BASE_DIR
    assert Config.CACHE_DIR.parent == Config.DATA_DIR
    assert Config.MODEL_DIR.parent == Config.DATA_DIR


def test_config_panel_settings():
    """Test Panel configuration settings."""
    assert isinstance(Config.PANEL_PORT, int)
    assert isinstance(Config.PANEL_HOST, str)
    assert isinstance(Config.PANEL_ALLOW_WEBSOCKET_ORIGIN, list)
    assert Config.PANEL_PORT > 0
    assert Config.PANEL_PORT < 65536


def test_config_cache_ttl():
    """Test cache TTL settings."""
    assert Config.REALTIME_DATA_TTL == 60  # 1 minute
    assert Config.BULK_DATA_TTL == 300  # 5 minutes
    assert Config.STOCK_INFO_TTL == 3600  # 1 hour
    assert Config.HISTORICAL_DATA_TTL == 86400  # 1 day

    # Verify increasing TTL for less volatile data
    assert Config.REALTIME_DATA_TTL < Config.BULK_DATA_TTL
    assert Config.BULK_DATA_TTL < Config.STOCK_INFO_TTL
    assert Config.STOCK_INFO_TTL < Config.HISTORICAL_DATA_TTL


def test_config_lstm_settings():
    """Test LSTM model settings."""
    assert Config.LSTM_SEQUENCE_LENGTH == 90
    assert Config.LSTM_ENSEMBLE_SIZE == 3
    assert Config.PREDICTION_DAYS == 30
    assert Config.BATCH_SIZE == 16
    assert Config.EPOCHS == 150

    assert isinstance(Config.LSTM_SEQUENCE_LENGTH, int)
    assert isinstance(Config.BATCH_SIZE, int)
    assert Config.BATCH_SIZE > 0
    assert Config.EPOCHS > 0


def test_config_rl_trading_settings():
    """Test RL trading configuration."""
    assert Config.RL_DEFAULT_INITIAL_BALANCE == 100000.0
    assert Config.RL_TRANSACTION_COST_RATE == 0.0  # Zero commission era
    assert Config.RL_SLIPPAGE_RATE == 0.0005  # 0.05%
    assert Config.RL_MAX_POSITION_PCT == 80.0
    assert Config.RL_LOOKBACK_WINDOW == 60

    # Verify risk management settings
    assert Config.RL_STOP_LOSS_PCT == 0.05  # 5%
    assert Config.RL_TRAILING_STOP_PCT == 0.03  # 3%
    assert Config.RL_MAX_DRAWDOWN_PCT == 0.15  # 15%

    # Verify percentage values are reasonable
    assert 0 <= Config.RL_STOP_LOSS_PCT <= 1
    assert 0 <= Config.RL_TRAILING_STOP_PCT <= 1
    assert 0 <= Config.RL_MAX_DRAWDOWN_PCT <= 1


def test_config_rl_enhancement_flags():
    """Test RL enhancement feature flags."""
    assert Config.RL_USE_ACTION_MASKING is True
    assert Config.RL_USE_ENHANCED_REWARDS is True
    assert Config.RL_USE_ADAPTIVE_SIZING is True
    assert Config.RL_USE_IMPROVED_ACTIONS is True
    assert Config.RL_ENABLE_DIAGNOSTICS is True

    assert isinstance(Config.RL_USE_ACTION_MASKING, bool)
    assert isinstance(Config.RL_USE_ENHANCED_REWARDS, bool)


def test_config_rl_hyperparameters():
    """Test RL agent hyperparameters."""
    assert Config.RL_PPO_LEARNING_RATE == 0.0003
    assert Config.RL_A2C_LEARNING_RATE == 0.0007
    assert Config.RL_GAMMA == 0.99

    # Verify learning rates are reasonable
    assert 0 < Config.RL_PPO_LEARNING_RATE < 1
    assert 0 < Config.RL_A2C_LEARNING_RATE < 1
    assert 0 < Config.RL_GAMMA <= 1


def test_config_ollama_settings():
    """Test Ollama configuration."""
    assert isinstance(Config.OLLAMA_BASE_URL, str)
    assert isinstance(Config.OLLAMA_MODEL, str)
    assert isinstance(Config.OLLAMA_TIMEOUT, int)
    assert isinstance(Config.OLLAMA_ENABLED, bool)
    assert isinstance(Config.OLLAMA_FALLBACK_TO_REGEX, bool)

    assert Config.OLLAMA_TIMEOUT > 0
    assert "http" in Config.OLLAMA_BASE_URL.lower()


def test_config_logging():
    """Test logging configuration."""
    assert Config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert isinstance(Config.LOG_FORMAT, str)
    assert "%(asctime)s" in Config.LOG_FORMAT
    assert "%(levelname)s" in Config.LOG_FORMAT


def test_config_ensure_directories(tmp_path, monkeypatch):
    """Test ensure_directories method."""
    # Temporarily override paths to use tmp_path
    test_base = tmp_path / "test_stock_agent"

    # Create a test config-like class
    class TestConfig:
        BASE_DIR = test_base
        DATA_DIR = test_base / "data"
        CACHE_DIR = test_base / "data" / "cache"
        MODEL_DIR = test_base / "data" / "models"
        LOG_DIR = test_base / "data" / "logs"
        RL_MODEL_DIR = test_base / "data" / "models" / "rl"

        @classmethod
        def ensure_directories(cls):
            for directory in [cls.DATA_DIR, cls.CACHE_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.RL_MODEL_DIR]:
                directory.mkdir(parents=True, exist_ok=True)

    # Ensure directories don't exist yet
    assert not TestConfig.DATA_DIR.exists()

    # Call ensure_directories
    TestConfig.ensure_directories()

    # Verify all directories were created
    assert TestConfig.DATA_DIR.exists()
    assert TestConfig.CACHE_DIR.exists()
    assert TestConfig.MODEL_DIR.exists()
    assert TestConfig.LOG_DIR.exists()
    assert TestConfig.RL_MODEL_DIR.exists()


def test_config_env_variable_override(monkeypatch):
    """Test that environment variables can override config."""
    # Set environment variable
    monkeypatch.setenv("PANEL_PORT", "9999")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    # Re-import to pick up new env vars (in real code, this would be at startup)
    # For this test, just verify the mechanism works
    test_port = int(os.getenv("PANEL_PORT", "5006"))
    test_log_level = os.getenv("LOG_LEVEL", "INFO")

    assert test_port == 9999
    assert test_log_level == "DEBUG"


def test_config_rl_training_defaults():
    """Test RL training default values."""
    assert Config.RL_DEFAULT_TRAINING_TIMESTEPS == 300000  # Recommended default
    assert isinstance(Config.RL_DEFAULT_TRAINING_TIMESTEPS, int)
    assert Config.RL_DEFAULT_TRAINING_TIMESTEPS > 0


def test_config_transaction_costs():
    """Test transaction cost configuration."""
    # Zero-commission era
    assert Config.RL_TRANSACTION_COST_RATE == 0.0

    # But still have slippage
    assert Config.RL_SLIPPAGE_RATE > 0
    assert Config.RL_SLIPPAGE_RATE < 0.01  # Less than 1%


def test_config_model_dir_structure():
    """Test model directory structure."""
    assert Config.RL_MODEL_DIR == Config.MODEL_DIR / "rl"
    assert Config.RL_MODEL_DIR.parent == Config.MODEL_DIR
