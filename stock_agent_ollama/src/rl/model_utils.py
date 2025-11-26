"""
Shared Model Loading Utilities

Provides unified model and configuration loading for training, backtesting,
and live trading.
"""

from pathlib import Path
from typing import Any, Optional, Dict
import json
import logging

logger = logging.getLogger(__name__)


def load_rl_agent(model_path: Path, env: Optional[Any] = None) -> Any:
    """
    Load RL agent with automatic type detection.

    This function automatically detects the agent type (PPO, RecurrentPPO, A2C, SAC, QRDQN)
    by reading the training config. It supports all 5 algorithms.

    Args:
        model_path: Path to model file (best_model.zip or final_model.zip)
        env: Optional environment to set (not needed for inference)

    Returns:
        Loaded agent with is_trained attribute set

    Raises:
        FileNotFoundError: If model file not found
        ValueError: If model cannot be loaded with any supported agent type
    """
    from stable_baselines3 import PPO, SAC, A2C
    from sb3_contrib import RecurrentPPO, QRDQN

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load training config to determine agent type
    config_path = model_path.parent / "training_config.json"
    agent_type = None

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                agent_type = config.get('agent_type', '').lower()
                logger.info(f"Detected agent type: {agent_type}")
        except Exception as e:
            logger.warning(f"Could not read training config: {e}")

    # Try loading with appropriate class based on config
    if agent_type == 'ppo':
        try:
            agent = PPO.load(str(model_path), env=env)
            agent.is_trained = True
            logger.info(f"Successfully loaded PPO from {model_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load PPO: {e}")
            raise

    elif agent_type == 'recurrent_ppo':
        try:
            agent = RecurrentPPO.load(str(model_path), env=env)
            agent.is_trained = True
            logger.info(f"Successfully loaded RecurrentPPO from {model_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load RecurrentPPO: {e}")
            raise

    elif agent_type == 'a2c':
        try:
            agent = A2C.load(str(model_path), env=env)
            agent.is_trained = True
            logger.info(f"Successfully loaded A2C from {model_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load A2C: {e}")
            raise

    elif agent_type == 'sac':
        try:
            # SAC uses continuous action space, need to wrap environment
            if env is not None:
                from .sac_discrete_wrapper import DiscreteToBoxWrapper
                env = DiscreteToBoxWrapper(env, n_discrete_actions=6)
            agent = SAC.load(str(model_path), env=env)
            agent.is_trained = True
            logger.info(f"Successfully loaded SAC from {model_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load SAC: {e}")
            raise

    elif agent_type == 'qrdqn':
        try:
            agent = QRDQN.load(str(model_path), env=env)
            agent.is_trained = True
            logger.info(f"Successfully loaded QRDQN from {model_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load QRDQN: {e}")
            raise

    else:
        # Auto-detect by trying each type
        logger.warning("Agent type not specified, attempting auto-detection...")
        for agent_name, agent_class in [
            ('PPO', PPO),
            ('RecurrentPPO', RecurrentPPO),
            ('A2C', A2C),
            ('SAC', SAC),
            ('QRDQN', QRDQN)
        ]:
            try:
                load_env = env
                # Special handling for SAC
                if agent_name == 'SAC' and env is not None:
                    from .sac_discrete_wrapper import DiscreteToBoxWrapper
                    load_env = DiscreteToBoxWrapper(env, n_discrete_actions=6)

                agent = agent_class.load(str(model_path), env=load_env)
                agent.is_trained = True
                logger.info(f"Successfully auto-detected and loaded {agent_name}")
                return agent
            except Exception:
                continue

        raise ValueError(f"Could not load model from {model_path} with any supported agent type")


def load_env_config_from_model(model_path: Path) -> Dict:
    """
    Load environment configuration from trained model directory.

    This allows live trading and backtesting to reproduce the exact
    environment configuration used during training.

    Args:
        model_path: Path to model file or directory

    Returns:
        Dictionary with environment configuration

    Raises:
        FileNotFoundError: If training config not found
    """
    model_path = Path(model_path)

    # If given a file, use its parent directory
    if model_path.is_file():
        config_dir = model_path.parent
    else:
        config_dir = model_path

    config_path = config_dir / "training_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {config_path}\n"
            f"Cannot load environment configuration from model."
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded environment config from {config_path}")
    return config


def save_env_config(config_dict: Dict, save_dir: Path) -> None:
    """
    Save environment configuration to model directory.

    Args:
        config_dict: Environment configuration dictionary
        save_dir: Directory to save config to
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "training_config.json"

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Saved environment config to {config_path}")
