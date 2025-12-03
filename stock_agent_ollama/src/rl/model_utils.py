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

    This function automatically detects the agent type (PPO, RecurrentPPO, Ensemble)
    by reading the training config. It supports policy gradient methods only.

    Args:
        model_path: Path to model file (best_model.zip or final_model.zip)
        env: Optional environment to set (not needed for inference)

    Returns:
        Loaded agent with is_trained attribute set

    Raises:
        FileNotFoundError: If model file not found
        ValueError: If model cannot be loaded with any supported agent type
    """
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from .ensemble import EnsemblePPOAgent

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

    elif agent_type == 'ensemble':
        try:
            # Load ensemble by loading both component models
            # Expect model_path to be the ensemble directory containing both models
            model_dir = model_path.parent if model_path.is_file() else model_path

            # Load PPO model
            ppo_path = model_dir / "ppo_best_model.zip"
            if not ppo_path.exists():
                # Try alternate path
                ppo_path = model_dir / "best_model_ppo.zip"

            ppo_model = PPO.load(str(ppo_path), env=env)

            # Load RecurrentPPO model
            rppo_path = model_dir / "recurrent_ppo_best_model.zip"
            if not rppo_path.exists():
                # Try alternate path
                rppo_path = model_dir / "best_model_recurrent_ppo.zip"

            rppo_model = RecurrentPPO.load(str(rppo_path), env=env)

            # Load ensemble config if available
            ensemble_config_path = model_dir / "ensemble_config.json"
            ppo_weight = 0.5
            rppo_weight = 0.5
            
            if ensemble_config_path.exists():
                try:
                    with open(ensemble_config_path, 'r') as f:
                        ens_config = json.load(f)
                        ppo_weight = ens_config.get('ppo_weight', 0.5)
                        rppo_weight = ens_config.get('recurrent_ppo_weight', 0.5)
                        logger.info(f"Loaded ensemble weights: PPO={ppo_weight}, RPPO={rppo_weight}")
                except Exception as e:
                    logger.warning(f"Failed to load ensemble config: {e}")

            # Create ensemble
            agent = EnsemblePPOAgent(
                ppo_model, 
                rppo_model,
                ppo_weight=ppo_weight,
                recurrent_ppo_weight=rppo_weight
            )
            agent.is_trained = True
            logger.info(f"Successfully loaded Ensemble from {model_dir}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load Ensemble: {e}")
            raise

    else:
        # Auto-detect by trying each type
        logger.warning("Agent type not specified, attempting auto-detection...")
        for agent_name, agent_class in [
            ('PPO', PPO),
            ('RecurrentPPO', RecurrentPPO)
        ]:
            try:
                agent = agent_class.load(str(model_path), env=env)
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
        # Try parent directory (common for ensemble where model is in subdirectory)
        parent_config_path = config_dir.parent / "training_config.json"
        if parent_config_path.exists():
            config_path = parent_config_path
        else:
            logger.warning(
                f"Training config not found: {config_path}\n"
                f"Using default EnvConfig values. Model may have been trained with different settings."
            )
            # Return default EnvConfig as dict
            from .env_factory import EnvConfig
            default_config = EnvConfig(
                symbol="UNKNOWN",
                start_date="2020-01-01",
                end_date="2023-12-31"
            )
            return default_config.to_dict()


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
