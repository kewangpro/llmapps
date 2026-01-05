import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.rl.env_factory import EnvConfig, create_enhanced_env
from src.rl.training import EnhancedTrainingConfig
from src.rl.ensemble import EnsemblePPOAgent
from src.rl.types import ImprovedTradingAction
from src.config import Config

# Mock Stable-Baselines3 models to avoid loading actual heavy models
class MockModel:
    def __init__(self, observation_space=None):
        self.observation_space = observation_space
        self.device = 'cpu'
        
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        # Return random action and state
        batch_size = observation.shape[0] if observation.ndim > 1 else 1
        return np.random.randint(0, 6, size=batch_size), None

@pytest.fixture
def mock_env_config():
    return EnvConfig(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_balance=10000.0
    )

def test_env_config_defaults():
    """Test that EnvConfig picks up defaults from Config correctly."""
    config = EnvConfig(symbol="TEST", start_date="2023-01-01", end_date="2023-01-10")
    
    assert config.initial_balance == Config.RL_DEFAULT_INITIAL_BALANCE
    assert config.transaction_cost_rate == Config.RL_TRANSACTION_COST_RATE
    assert config.slippage_rate == Config.RL_SLIPPAGE_RATE
    assert config.max_position_pct == Config.RL_MAX_POSITION_PCT
    assert config.lookback_window == Config.RL_LOOKBACK_WINDOW
    assert config.stop_loss_pct == Config.RL_STOP_LOSS_PCT

def test_training_config_defaults():
    """Test that EnhancedTrainingConfig picks up defaults from Config."""
    config = EnhancedTrainingConfig(symbol="TEST", start_date="2023-01-01", end_date="2023-01-10")
    
    assert config.initial_balance == Config.RL_DEFAULT_INITIAL_BALANCE
    assert config.max_position_pct == Config.RL_MAX_POSITION_PCT
    assert config.lookback_window == Config.RL_LOOKBACK_WINDOW
    assert config.total_timesteps == Config.RL_DEFAULT_TRAINING_TIMESTEPS

def test_ensemble_initialization():
    """Test initialization of EnsemblePPOAgent."""
    ppo_mock = MockModel()
    rppo_mock = MockModel()
    
    ensemble = EnsemblePPOAgent(
        ppo_model=ppo_mock,
        recurrent_ppo_model=rppo_mock,
        ppo_weight=0.4,
        recurrent_ppo_weight=0.6,
        use_confidence=False
    )
    
    # Weights should be normalized (already are sum=1, but checking logic)
    assert ensemble.ppo_weight == 0.4
    assert ensemble.recurrent_ppo_weight == 0.6
    
    # Test normalization
    ensemble_norm = EnsemblePPOAgent(
        ppo_model=ppo_mock,
        recurrent_ppo_model=rppo_mock,
        ppo_weight=2.0,
        recurrent_ppo_weight=8.0,
        use_confidence=False
    )
    assert ensemble_norm.ppo_weight == 0.2
    assert ensemble_norm.recurrent_ppo_weight == 0.8

def test_ensemble_voting_logic():
    """Test the voting logic of the ensemble agent."""
    # Mock models that always return specific actions
    class FixedActionModel:
        def __init__(self, action):
            self.action = action
            self.device = 'cpu'
            self.observation_space = MagicMock()
            self.observation_space.shape = (10,)
            
        def predict(self, observation, state=None, episode_start=None, deterministic=True):
            return np.array([self.action]), None

    # Scenario 1: Agreement
    ensemble = EnsemblePPOAgent(
        ppo_model=FixedActionModel(1),
        recurrent_ppo_model=FixedActionModel(1),
        use_confidence=False
    )
    obs = np.zeros((1, 10))
    action, _ = ensemble.predict(obs)
    assert action[0] == 1
    assert ensemble.agreement_count == 1

    # Scenario 2: Disagreement (Weighted Vote)
    # We can't deterministic test random.random() easily without seeding, 
    # but we can test the Conflict Resolution logic in _weighted_vote_with_confidence
    
    # Scenario 3: Conflict Resolution (Buy vs Sell -> HOLD)
    # This logic exists in _weighted_vote_with_confidence
    ensemble_conf = EnsemblePPOAgent(
        ppo_model=FixedActionModel(int(ImprovedTradingAction.BUY_LARGE)), # Action 3
        recurrent_ppo_model=FixedActionModel(int(ImprovedTradingAction.SELL_ALL)), # Action 5
        use_confidence=True
    )
    
    # Mock _get_action_probabilities to return dummy probs
    ensemble_conf._get_action_probabilities = MagicMock(return_value=np.ones(6)/6)
    
    action, _ = ensemble_conf.predict(obs)
    # Opposing actions (Buy vs Sell) should result in HOLD (0)
    assert action[0] == int(ImprovedTradingAction.HOLD) 

@patch('src.tools.stock_fetcher.StockFetcher.fetch_stock_data')
def test_env_creation(mock_fetch, mock_env_config):
    """Test that environment can be created from config."""
    # Mock data return
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = pd.DataFrame({
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    mock_fetch.return_value = data
    
    env = create_enhanced_env(mock_env_config)
    
    assert env is not None
    assert env.symbol == "TEST"
    assert env.initial_balance == 10000.0
    assert env.lookback_window == Config.RL_LOOKBACK_WINDOW
