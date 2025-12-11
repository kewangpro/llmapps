"""
Shared Model Loading Utilities

Provides unified model and configuration loading for training, backtesting,
and live trading.
"""

from pathlib import Path
from typing import Any, Optional, Dict, Tuple, Union
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def normalize_model_path(model_path: Union[str, Path], agent_type: Optional[str] = None) -> Path:
    """
    Normalize model path to the correct format for loading.

    This is the single source of truth for path resolution across the system.

    Args:
        model_path: Path to model directory or file
        agent_type: Optional agent type ('ppo', 'recurrent_ppo', 'ensemble')
                   If not provided, will be auto-detected from training_config.json

    Returns:
        Path: Normalized path ready for load_rl_agent()
              - For PPO/RecurrentPPO: path to best_model.zip or final_model.zip
              - For Ensemble: path to ensemble subdirectory containing component models

    Raises:
        FileNotFoundError: If model path or required files don't exist
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # If already a file, return it (already normalized)
    if model_path.is_file():
        return model_path

    # It's a directory - need to determine what kind of model
    model_dir = model_path

    # Check if this is an ensemble directory by detecting component models
    # Do this BEFORE agent_type detection to handle ensemble subdirectories
    is_ensemble_dir = (model_dir / "ppo_best_model.zip").exists() and \
                      (model_dir / "recurrent_ppo_best_model.zip").exists()

    if is_ensemble_dir:
        logger.debug(f"Detected ensemble directory by component models: {model_dir}")
        return model_dir

    # Auto-detect agent type if not provided
    if agent_type is None:
        # Try current directory first
        config_path = model_dir / "training_config.json"

        # If not found, try parent directory (for ensemble subdirectories)
        if not config_path.exists() and model_dir.name == "ensemble":
            config_path = model_dir.parent / "training_config.json"

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    agent_type = config.get('agent_type', '').lower()
                    logger.debug(f"Auto-detected agent type: {agent_type}")
            except Exception as e:
                logger.warning(f"Could not read training config: {e}")

    # Handle ensemble models
    if agent_type == 'ensemble':
        # Ensemble models are in an 'ensemble' subdirectory
        ensemble_dir = model_dir / "ensemble"

        # Check if ensemble subdirectory exists
        if ensemble_dir.exists():
            logger.debug(f"Found ensemble subdirectory: {ensemble_dir}")
            return ensemble_dir

        # Fallback: return model_dir and let load_rl_agent handle it
        logger.warning(f"Ensemble directory structure not found, using: {model_dir}")
        return model_dir

    # Handle PPO and RecurrentPPO models
    else:
        # Look for best_model.zip first, then final_model.zip
        model_file = model_dir / "best_model.zip"
        if model_file.exists():
            logger.debug(f"Found best_model.zip: {model_file}")
            return model_file

        model_file = model_dir / "final_model.zip"
        if model_file.exists():
            logger.debug(f"Found final_model.zip: {model_file}")
            return model_file

        raise FileNotFoundError(f"No model file (best_model.zip or final_model.zip) found in {model_dir}")


class CompatibilityAgent:
    """
    Wrapper for SB3 agents to handle observation mismatches (e.g., extra features like masks/trend).
    Dynamically slices input observations to match the agent's expected observation space.
    """
    def __init__(self, agent: Any):
        self.agent = agent

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict action with observation adjustment.
        """
        if hasattr(self.agent, 'observation_space'):
            expected_features = self.agent.observation_space.shape[-1]
            input_features = observation.shape[-1]

            if input_features > expected_features:
                # Slicing logic (matches EnsemblePPOAgent logic)
                
                # Case 1: Input has Trend(3) but Agent doesn't want it (e.g., 32 -> 29)
                if input_features == expected_features + 3:
                    try:
                        if observation.ndim > 1:
                            observation = np.concatenate([observation[..., :10], observation[..., 13:]], axis=-1)
                        else:
                            observation = np.concatenate([observation[:10], observation[13:]])
                    except Exception:
                        pass # Fallback to full obs if slice fails

                # Case 2: Input has Trend(3) + Mask(6) but Agent wants neither (32 -> 23)
                elif input_features == 32 and expected_features == 23:
                    try:
                        if observation.ndim > 1:
                            observation = np.concatenate([observation[..., :10], observation[..., 13:26]], axis=-1)
                        else:
                            observation = np.concatenate([observation[:10], observation[13:26]])
                    except Exception:
                        pass

                # Case 3: Input has Mask(6) but Agent doesn't want it (e.g., 29 -> 23 or 32 -> 26)
                elif input_features == expected_features + 6:
                    try:
                        if observation.ndim > 1:
                            observation = observation[..., :-6]
                        else:
                            observation = observation[:-6]
                    except Exception:
                        pass
        
        return self.agent.predict(observation, state, episode_start, deterministic)


def load_vec_normalize(model_path: Path, env: Any) -> Any:
    """
    Load VecNormalize statistics and wrap environment if available.

    Args:
        model_path: Path to model file or directory
        env: The environment to wrap

    Returns:
        Wrapped environment (VecNormalize) or original environment
    """
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    model_path = Path(model_path)
    if model_path.is_file():
        model_dir = model_path.parent
    else:
        model_dir = model_path

    # Check for vec_normalize.pkl
    vec_path = model_dir / "vec_normalize.pkl"
    
    # Check in parent if not found (for ensemble subdirs)
    if not vec_path.exists():
        vec_path = model_dir.parent / "vec_normalize.pkl"

    if vec_path.exists():
        try:
            # VecNormalize requires a VecEnv
            if not hasattr(env, 'reset'): # Basic check, ideally isinstance(env, VecEnv)
                 # If it's a Gym env, wrap in DummyVecEnv
                 env = DummyVecEnv([lambda: env])
            
            env = VecNormalize.load(str(vec_path), env)
            env.training = False  # Disable updating stats during inference
            env.norm_reward = False  # Disable reward normalization during inference
            logger.info(f"Loaded VecNormalize stats from {vec_path}")
        except Exception as e:
            logger.warning(f"Failed to load VecNormalize stats: {e}")
    
    return env


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

    # Determine model directory
    if model_path.is_file():
        model_dir = model_path.parent
    else:
        model_dir = model_path

    # Load VecNormalize stats if available and env provided
    if env is not None:
        env = load_vec_normalize(model_path, env)

    # Load training config to determine agent type
    # Config is always in the top-level model directory, not in subdirectories
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        # If not, check parent (might be in a subdirectory like 'ensemble')
        config_path = model_dir.parent / "training_config.json"

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
            agent = CompatibilityAgent(agent)  # Wrap for observation compatibility
            agent.is_trained = True
            logger.info(f"Successfully loaded PPO from {model_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load PPO: {e}")
            raise

    elif agent_type == 'recurrent_ppo':
        try:
            agent = RecurrentPPO.load(str(model_path), env=env)
            agent = CompatibilityAgent(agent)  # Wrap for observation compatibility
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
            
            # Check if models are in an 'ensemble' subdirectory (but not if we're already in it)
            if model_dir.name != 'ensemble':
                ensemble_subdir = model_dir / "ensemble"
                if ensemble_subdir.exists():
                    model_dir = ensemble_subdir

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
                agent = CompatibilityAgent(agent)  # Wrap for compatibility
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