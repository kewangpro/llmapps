"""
Enhanced Training Script with All Improvements

Usage:
    python -m src.rl.enhanced_training --symbol GOOGL --use-improvements

This script provides comprehensive training improvements including:
- Action masking
- Enhanced rewards
- Curriculum learning
- Better diagnostics
"""

from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from sb3_contrib import RecurrentPPO

from .environments import EnhancedTradingEnv
from .improvements import (
    EnhancedRewardConfig,
    PPORewardConfig,
    RecurrentPPORewardConfig,
    CurriculumManager,
    TrainingDiagnostics
)
from .callbacks import TrainingProgressCallback, PerformanceMonitorCallback
from .env_factory import EnvConfig

logger = logging.getLogger(__name__)

# Reference defaults from EnvConfig (single source of truth)
_ENV_DEFAULTS = {f.name: f.default for f in EnvConfig.__dataclass_fields__.values()}


@dataclass
class EnhancedTrainingConfig:
    """Configuration for enhanced RL training."""

    # Basic settings
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = _ENV_DEFAULTS['initial_balance']

    # Environment enhancements
    use_action_masking: bool = _ENV_DEFAULTS['use_action_masking']
    use_enhanced_rewards: bool = _ENV_DEFAULTS['use_enhanced_rewards']
    use_adaptive_sizing: bool = _ENV_DEFAULTS['use_adaptive_sizing']
    use_improved_actions: bool = _ENV_DEFAULTS['use_improved_actions']
    # Disabled curriculum learning - was causing excessive exploration leading to 65% invalid action rate
    # Agent performs better with direct learning on the full task
    use_curriculum_learning: bool = False
    enable_diagnostics: bool = _ENV_DEFAULTS['enable_diagnostics']
    use_lstm_policy: bool = False # New field to enable LSTM policy

    # New improvements (Priority 1-5)
    use_risk_manager: bool = True
    use_regime_detector: bool = True
    use_mtf_features: bool = True
    use_kelly_sizing: bool = True
    stop_loss_pct: float = 0.05
    trailing_stop_pct: float = 0.03
    max_drawdown_pct: float = 0.15

    # Position limits (from EnvConfig)
    max_position_size: int = _ENV_DEFAULTS['max_position_size']
    max_position_pct: float = _ENV_DEFAULTS['max_position_pct']

    # Observation parameters (from EnvConfig)
    lookback_window: int = _ENV_DEFAULTS['lookback_window']

    # Reward configuration
    reward_config: EnhancedRewardConfig = field(default_factory=EnhancedRewardConfig)

    # Agent settings
    agent_type: str = "ppo"  # Options: 'ppo', 'recurrent_ppo', 'ensemble'
    learning_rate: float = 3e-4
    gamma: float = 0.99
    # Entropy coefficient controls exploration vs exploitation
    # 0.05 caused action collapse (agent only used BUY_MEDIUM)
    # 0.1 still showed action collapse (86% BUY_SMALL, 14% BUY_MEDIUM)
    # 0.2 provides stronger exploration to prevent collapse to 1-2 actions
    ent_coef: float = 0.2
    n_steps: int = 2048  # PPO: steps per update
    batch_size: int = 128  # Increased for better gradient estimates
    n_epochs: int = 10

    # Training settings
    total_timesteps: int = 300000  # Recommended for all algorithms, especially RecurrentPPO (LSTM needs more training)
    eval_freq: int = 5000
    save_freq: int = 10000

    # Transaction costs (from EnvConfig)
    # Algorithm-specific transaction costs via reward configs:
    # - QRDQN: 0.0005 (0.05%) via EnhancedRewardConfig
    # - SAC: 0.001 (0.1%) via SACRewardConfig
    # - PPO: 0.002 (0.2%) via PPORewardConfig
    # - RecurrentPPO: 0.001 (0.1%) via RecurrentPPORewardConfig
    transaction_cost_rate: float = _ENV_DEFAULTS['transaction_cost_rate']  # 0.05% per trade (base)
    slippage_rate: float = _ENV_DEFAULTS['slippage_rate']

    # Save settings
    save_dir: Optional[str] = None
    verbose: int = 1


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning progression.
    """

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        check_freq: int = 1000,
        advancement_threshold: float = 0.0,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.check_freq = check_freq
        self.advancement_threshold = advancement_threshold
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        """Check curriculum advancement."""
        # Track episode rewards
        self.current_episode_reward += self.locals.get('rewards', [0])[0]

        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            self.curriculum_manager.on_episode_end(self.episode_rewards[-1])

        # Check for advancement
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards[-100:])

            if self.curriculum_manager.should_advance(mean_reward, self.advancement_threshold):
                old_stage = self.curriculum_manager.get_current_stage().name
                self.curriculum_manager.advance_stage()
                new_stage = self.curriculum_manager.get_current_stage().name

                if self.verbose > 0:
                    print(f"\n🎓 Curriculum Advanced: {old_stage} → {new_stage}")
                    print(f"   Mean Reward: {mean_reward:.4f}")
                    print(f"   Episodes in stage: {self.curriculum_manager.episode_count}\n")

        return True


class DiagnosticsCallback(BaseCallback):
    """
    Callback for collecting and reporting training diagnostics.
    """

    def __init__(
        self,
        env: EnhancedTradingEnv,
        report_freq: int = 10000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.env = env
        self.report_freq = report_freq

    def _on_step(self) -> bool:
        """Periodically report diagnostics."""
        if self.n_calls % self.report_freq == 0 and self.verbose > 0:
            if hasattr(self.env, 'print_diagnostics'):
                print(f"\n📊 Training Diagnostics (Step {self.num_timesteps}):")
                self.env.print_diagnostics()

        return True


class EnhancedRLTrainer:
    """
    Enhanced RL trainer with all improvements.
    """

    def __init__(self, config: EnhancedTrainingConfig, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.env = None
        self.agent = None
        self.curriculum_manager = None
        self.callbacks = []

        # Select appropriate reward config based on agent type
        # Algorithm-specific reward configurations:
        # - PPO: PPORewardConfig (stronger penalties to fight action collapse)
        # - RecurrentPPO: RecurrentPPORewardConfig (reduced penalties for trend riding)
        if config.reward_config is None or isinstance(config.reward_config, EnhancedRewardConfig):
            if config.agent_type.lower() == 'recurrent_ppo':
                logger.info(f"Using RecurrentPPORewardConfig for RecurrentPPO (HOLD incentive + momentum bonus)")
                self.config.reward_config = RecurrentPPORewardConfig()
            elif config.agent_type.lower() == 'ppo':
                logger.info(f"Using PPORewardConfig for PPO")
                self.config.reward_config = PPORewardConfig()
            else:  # Unknown agent type or ensemble
                logger.info(f"Using EnhancedRewardConfig for {config.agent_type.upper()}")
                if config.reward_config is None:
                    self.config.reward_config = EnhancedRewardConfig()

        # Setup save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{config.agent_type}_{config.symbol}_{timestamp}"
            self.save_dir = Path(f"data/models/rl/{model_name}")

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def setup_environment(self) -> EnhancedTradingEnv:
        """Create enhanced trading environment using shared factory."""
        from .env_factory import EnvConfig, create_enhanced_env

        # Setup curriculum learning
        if self.config.use_curriculum_learning:
            self.curriculum_manager = CurriculumManager()
            logger.info(f"Curriculum learning enabled. Starting at stage: {self.curriculum_manager.get_current_stage().name}")
        else:
            self.curriculum_manager = None

        # Build environment config
        # Enable trend indicators for RecurrentPPO (for trend-following capability)
        include_trend = self.config.agent_type.lower() == 'recurrent_ppo'
        if include_trend:
            logger.info("Enabling trend indicators for RecurrentPPO")

        env_config = EnvConfig(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_balance=self.config.initial_balance,
            transaction_cost_rate=self.config.transaction_cost_rate,
            slippage_rate=self.config.slippage_rate,
            max_position_size=self.config.max_position_size,
            max_position_pct=self.config.max_position_pct,
            lookback_window=self.config.lookback_window,
            include_technical_indicators=True,
            include_trend_indicators=include_trend,  # Enable for LSTM PPO
            use_action_masking=self.config.use_action_masking,
            use_enhanced_rewards=self.config.use_enhanced_rewards,
            use_adaptive_sizing=self.config.use_adaptive_sizing,
            use_improved_actions=self.config.use_improved_actions,
            reward_config=self.config.reward_config,
            curriculum_manager=self.curriculum_manager,
            enable_diagnostics=self.config.enable_diagnostics,
            # New improvements
            use_risk_manager=self.config.use_risk_manager,
            use_regime_detector=self.config.use_regime_detector,
            use_mtf_features=self.config.use_mtf_features,
            use_kelly_sizing=self.config.use_kelly_sizing,
            stop_loss_pct=self.config.stop_loss_pct,
            trailing_stop_pct=self.config.trailing_stop_pct,
            max_drawdown_pct=self.config.max_drawdown_pct
        )

        # Create environment using shared factory
        self.env = create_enhanced_env(env_config)
        return self.env

    def setup_agent(self) -> Any:
        """Create RL agent."""
        if self.env is None:
            self.setup_environment()

        # Wrap environment in DummyVecEnv and then VecNormalize
        # This is crucial for PPO and RecurrentPPO
        # Only do this if not already wrapped
        if not isinstance(self.env, DummyVecEnv) and self.config.agent_type.lower() != 'ensemble':
            env = self.env  # Capture env for lambda
            self.env = DummyVecEnv([lambda: env]) # Wrap in DummyVecEnv first
            self.env = VecNormalize(
                self.env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=self.config.gamma
            )
            logger.info("Wrapped environment with DummyVecEnv and VecNormalize (norm_obs=True, norm_reward=True)")
        elif self.config.agent_type.lower() == 'ensemble':
            # For ensemble, the individual PPO/RPPO trainers will handle their own VecNormalize
            # The env for the main trainer will remain unwrapped if not a single agent
            logger.info("Ensemble mode: Skipping VecNormalize wrapping for main env. Sub-trainers will handle.")


        # Determine agent type for algorithm-specific settings
        agent_type_lower = self.config.agent_type.lower()

        # Common agent parameters
        agent_params = {
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'verbose': self.config.verbose,
        }

        # Determine agent class and policy type
        agent_class = None
        policy_type = 'MlpPolicy' # Default policy

        if agent_type_lower == 'ppo':
            agent_class = PPO
            policy_type = 'MlpPolicy'
        elif agent_type_lower == 'recurrent_ppo':
            agent_class = RecurrentPPO
            policy_type = 'MlpLstmPolicy'
        elif agent_type_lower == 'ensemble':
            # Ensemble will be handled separately after training both base models
            agent_class = None
            policy_type = None
        else:
            raise ValueError(f"Unsupported agent type: {self.config.agent_type}. Available: ppo, recurrent_ppo, ensemble")

        # Algorithm-specific parameters
        if agent_type_lower in ['ppo', 'recurrent_ppo']:
            agent_params.update({
                'n_steps': self.config.n_steps,
                'batch_size': self.config.batch_size,
                'n_epochs': self.config.n_epochs,
                'ent_coef': self.config.ent_coef,
            })

        # Create agent (ensemble is handled in train() method)
        if agent_type_lower != 'ensemble':
            self.agent = agent_class(
                policy_type,
                self.env,
                **agent_params
            )
            logger.info(f"Created {self.config.agent_type.upper()} agent with {policy_type}")
        else:
            # Ensemble: will train both PPO and RecurrentPPO, then combine
            self.agent = None
            logger.info("Ensemble mode: will train both PPO and RecurrentPPO")

        return self.agent

    def setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = []

        # Progress tracking
        self.callbacks.append(TrainingProgressCallback(
            check_freq=1000,
            save_freq=self.config.save_freq,
            save_path=self.save_dir / "checkpoints",
            progress_callback=self.progress_callback,
            verbose=self.config.verbose
        ))

        # Best model saving
        self.callbacks.append(PerformanceMonitorCallback(
            save_path=self.save_dir / "best_model.zip",
            check_freq=1000,
            verbose=self.config.verbose
        ))

        # Curriculum learning
        if self.config.use_curriculum_learning and self.curriculum_manager:
            self.callbacks.append(CurriculumCallback(
                curriculum_manager=self.curriculum_manager,
                check_freq=1000,
                advancement_threshold=0.0,
                verbose=self.config.verbose
            ))

        # Diagnostics (only works if env has print_diagnostics method)
        # Note: VecNormalize wraps the env, so we need to access the underlying env
        if self.config.enable_diagnostics:
            # Unpack env if wrapped
            actual_env = self.env
            if isinstance(actual_env, VecNormalize):
                actual_env = actual_env.venv
            # DummyVecEnv wraps the actual env
            if hasattr(actual_env, 'envs'):
                actual_env = actual_env.envs[0]
            
            self.callbacks.append(DiagnosticsCallback(
                env=actual_env,
                report_freq=10000,
                verbose=self.config.verbose
            ))

    def train(self) -> Dict[str, Any]:
        """Execute enhanced training."""

        if self.env is None:
            self.setup_environment()

        # Check if ensemble mode
        if self.config.agent_type.lower() == 'ensemble':
            return self._train_ensemble()

        if self.agent is None:
            self.setup_agent()

        self.setup_callbacks()

        # Print training configuration
        if self.config.verbose > 0:
            self._print_training_info()

        # Train
        start_time = datetime.now()

        self.agent.learn(
            total_timesteps=self.config.total_timesteps,
            callback=self.callbacks,
            progress_bar=True
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Save final model
        final_model_path = self.save_dir / "final_model.zip"
        self.agent.save(str(final_model_path))
        
        # Save VecNormalize statistics if used
        if isinstance(self.env, VecNormalize):
            vec_normalize_path = self.save_dir / "vec_normalize.pkl"
            self.env.save(str(vec_normalize_path))
            logger.info(f"Saved VecNormalize statistics to {vec_normalize_path}")

        # Save configuration
        self.save_config()

        # Get diagnostics (access underlying env)
        diagnostics = {}
        if self.config.enable_diagnostics:
            actual_env = self.env
            if isinstance(actual_env, VecNormalize):
                actual_env = actual_env.venv
            if hasattr(actual_env, 'envs'):
                actual_env = actual_env.envs[0]
            
            if hasattr(actual_env, 'get_diagnostics_summary'):
                diagnostics = actual_env.get_diagnostics_summary()

        # Get training stats from progress callback
        progress_callback = None
        for callback in self.callbacks:
            if isinstance(callback, TrainingProgressCallback):
                progress_callback = callback
                break

        training_stats = progress_callback.get_training_stats() if progress_callback else {}

        # Extract explained variance from model logger (PPO specific)
        explained_variance = None
        try:
            if hasattr(self.agent, 'logger') and self.agent.logger is not None:
                # Get the last logged value of explained variance
                if hasattr(self.agent.logger, 'name_to_value'):
                    explained_variance = self.agent.logger.name_to_value.get('train/explained_variance')
        except Exception as e:
            logger.debug(f"Could not extract explained variance: {e}")

        # Compile results
        results = {
            'success': True,
            'training_time': training_time,
            'total_timesteps': self.config.total_timesteps,
            'total_episodes': training_stats.get('total_episodes', 0),
            'final_model_path': str(final_model_path),
            'save_dir': str(self.save_dir),
            'training_stats': training_stats,
            'diagnostics': diagnostics,
            'explained_variance': explained_variance,
            'improvements_used': {
                'action_masking': self.config.use_action_masking,
                'enhanced_rewards': self.config.use_enhanced_rewards,
                'adaptive_sizing': self.config.use_adaptive_sizing,
                'improved_actions': self.config.use_improved_actions,
                'curriculum_learning': self.config.use_curriculum_learning,
                'vec_normalize': isinstance(self.env, VecNormalize)
            }
        }

        if self.config.verbose > 0:
            self._print_training_summary(results)

        return results

    def _train_ensemble(self) -> Dict[str, Any]:
        """Train ensemble by training both PPO and RecurrentPPO, then combining them."""
        from .ensemble import EnsemblePPOAgent

        print(f"\n{'='*70}")
        print("🚀 ENSEMBLE TRAINING: PPO + RecurrentPPO")
        print(f"{'='*70}")
        print(f"Stock: {self.config.symbol}")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Total Timesteps per agent: {self.config.total_timesteps:,}")
        print(f"\nStrategy:")
        print(f"  1. Train PPO (aggressive growth)")
        print(f"  2. Train RecurrentPPO (risk management)")
        print(f"  3. Combine with 60/40 weighted voting")
        print(f"{'='*70}\n")

        ensemble_start_time = datetime.now()

        # Train PPO
        print("\n" + "="*70)
        print("Step 1/2: Training PPO Agent")
        print("="*70)

        # Create a copy of config for PPO to avoid modifying the ensemble config
        from dataclasses import replace
        ppo_config = replace(self.config, agent_type='ppo')
        ppo_trainer = EnhancedRLTrainer(ppo_config, progress_callback=self.progress_callback)
        ppo_results = ppo_trainer.train()

        ppo_model_path = ppo_results['final_model_path']
        # Load without env to preserve the model's original observation space
        # Note: We need to handle VecNormalize separately for ensemble
        ppo_model = PPO.load(ppo_model_path)

        print(f"✅ PPO training complete: {ppo_results['training_time']:.1f}s")

        # Train RecurrentPPO
        print("\n" + "="*70)
        print("Step 2/2: Training RecurrentPPO Agent")
        print("="*70)

        # Create a copy of config for RecurrentPPO to avoid modifying the ensemble config
        rppo_config = replace(self.config, agent_type='recurrent_ppo')
        rppo_trainer = EnhancedRLTrainer(rppo_config, progress_callback=self.progress_callback)
        rppo_results = rppo_trainer.train()

        rppo_model_path = rppo_results['final_model_path']
        # Load without env to preserve the model's original observation space
        rppo_model = RecurrentPPO.load(rppo_model_path)

        print(f"✅ RecurrentPPO training complete: {rppo_results['training_time']:.1f}s")

        # Create ensemble
        print("\n" + "="*70)
        print("Creating Ensemble Agent")
        print("="*70)

        ensemble = EnsemblePPOAgent(
            ppo_model=ppo_model,
            recurrent_ppo_model=rppo_model,
            ppo_weight=0.3,
            recurrent_ppo_weight=0.7
        )

        # Save ensemble models to ensemble directory
        ensemble_dir = self.save_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Copy models AND VecNormalize stats to ensemble directory
        import shutil
        shutil.copy(ppo_model_path, ensemble_dir / "ppo_best_model.zip")
        shutil.copy(rppo_model_path, ensemble_dir / "recurrent_ppo_best_model.zip")
        
        # Copy vec_normalize stats if they exist
        ppo_vec_path = Path(ppo_results['save_dir']) / "vec_normalize.pkl"
        if ppo_vec_path.exists():
            shutil.copy(ppo_vec_path, ensemble_dir / "ppo_vec_normalize.pkl")
            
        rppo_vec_path = Path(rppo_results['save_dir']) / "vec_normalize.pkl"
        if rppo_vec_path.exists():
            shutil.copy(rppo_vec_path, ensemble_dir / "recurrent_ppo_vec_normalize.pkl")

        # Save ensemble config
        ensemble.save(ensemble_dir)

        total_training_time = (datetime.now() - ensemble_start_time).total_seconds()

        print(f"\n✅ Ensemble created successfully!")
        print(f"  PPO Weight: 30%")
        print(f"  RecurrentPPO Weight: 70%")
        print(f"  Total Training Time: {total_training_time:.1f}s")
        print(f"  Saved to: {ensemble_dir}")

        # Save ensemble configuration
        self.save_config()

        # Compile results
        results = {
            'success': True,
            'training_time': total_training_time,
            'total_timesteps': self.config.total_timesteps * 2,  # Both agents
            'total_episodes': ppo_results.get('total_episodes', 0) + rppo_results.get('total_episodes', 0),
            'ppo_results': ppo_results,
            'recurrent_ppo_results': rppo_results,
            'ensemble_dir': str(ensemble_dir),
            'final_model_path': str(ensemble_dir),  # Point to ensemble directory for model loading
            'ppo_model_path': str(ensemble_dir / "ppo_best_model.zip"),
            'recurrent_ppo_model_path': str(ensemble_dir / "recurrent_ppo_best_model.zip"),
            'improvements_used': {
                'action_masking': self.config.use_action_masking,
                'enhanced_rewards': self.config.use_enhanced_rewards,
                'adaptive_sizing': self.config.use_adaptive_sizing,
                'improved_actions': self.config.use_improved_actions,
                'vec_normalize': True
            }
        }

        return results

    def _print_training_info(self):
        """Print training configuration."""
        print(f"\n{'='*70}")
        print("🚀 ENHANCED RL TRAINING")
        print(f"{'='*70}")
        print(f"Stock: {self.config.symbol}")
        print(f"Agent: {self.config.agent_type.upper()}")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Total Timesteps: {self.config.total_timesteps:,}")

        print(f"\n📈 Improvements Enabled:")
        print(f"  ✓ Action Masking: {self.config.use_action_masking}")
        print(f"  ✓ Enhanced Rewards: {self.config.use_enhanced_rewards}")
        print(f"  ✓ Adaptive Sizing: {self.config.use_adaptive_sizing}")
        print(f"  ✓ Improved Actions (6-action): {self.config.use_improved_actions}")
        print(f"  ✓ Curriculum Learning: {self.config.use_curriculum_learning}")
        print(f"  ✓ Training Diagnostics: {self.config.enable_diagnostics}")
        print(f"  ✓ VecNormalize: True (Obs & Reward)")

        if self.config.use_curriculum_learning:
            print(f"\n🎓 Curriculum Stages:")
            for i, stage in enumerate(self.curriculum_manager.stages):
                marker = "→" if i == 0 else " "
                print(f"  {marker} Stage {i+1}: {stage.name} - {stage.description}")

        print(f"\n💾 Save Directory: {self.save_dir}")
        print(f"{'='*70}\n")

    def _print_training_summary(self, results: Dict[str, Any]):
        """Print training summary."""
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Time: {results['training_time']:.1f}s ({results['training_time']/60:.1f} min)")
        print(f"Model saved: {results['final_model_path']}")

        if results.get('diagnostics'):
            diag = results['diagnostics']
            print(f"\n📊 Final Diagnostics:")
            print(f"  Total Episodes: {diag.get('total_episodes', 'N/A')}")
            print(f"  Invalid Action Rate: {diag.get('invalid_action_rate', 0):.2%}")
            print(f"  Mean Episode Reward: {diag.get('mean_episode_reward', 0):.4f}")
            print(f"  Mean Portfolio Return: {diag.get('mean_portfolio_return', 0):.2%}")

        print(f"{'='*70}\n")

    def save_config(self):
        """Save complete training configuration for reproducibility."""
        config_path = self.save_dir / "training_config.json"

        config_dict = {
            # Environment parameters - needed to reproduce exact training environment
            'symbol': self.config.symbol,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'initial_balance': self.config.initial_balance,

            # Cost parameters - CRITICAL for matching training conditions
            'transaction_cost_rate': self.config.transaction_cost_rate,
            'slippage_rate': self.config.slippage_rate,

            # Position limits
            'max_position_size': self.config.max_position_size,
            'max_position_pct': self.config.max_position_pct,

            # Enhancement flags - needed to recreate environment
            'use_action_masking': self.config.use_action_masking,
            'use_enhanced_rewards': self.config.use_enhanced_rewards,
            'use_adaptive_sizing': self.config.use_adaptive_sizing,
            'use_improved_actions': self.config.use_improved_actions,
            'use_curriculum_learning': self.config.use_curriculum_learning,
            'use_lstm_policy': self.config.use_lstm_policy,
            'use_vec_normalize': True, # Record usage of VecNormalize

            # New improvement flags (Priority 1-5)
            'use_risk_manager': getattr(self.config, 'use_risk_manager', True),
            'use_regime_detector': getattr(self.config, 'use_regime_detector', True),
            'use_mtf_features': getattr(self.config, 'use_mtf_features', True),
            'use_kelly_sizing': getattr(self.config, 'use_kelly_sizing', True),
            'stop_loss_pct': getattr(self.config, 'stop_loss_pct', 0.05),
            'trailing_stop_pct': getattr(self.config, 'trailing_stop_pct', 0.03),
            'max_drawdown_pct': getattr(self.config, 'max_drawdown_pct', 0.15),

            # Agent settings
            'agent_type': self.config.agent_type,
            'learning_rate': self.config.learning_rate,
            'total_timesteps': self.config.total_timesteps,

            # Observation space
            'lookback_window': self.config.lookback_window,
            'include_technical_indicators': True,  # Always true in current implementation
            'include_trend_indicators': self.config.agent_type.lower() == 'recurrent_ppo',

            # Optional flags
            'enable_diagnostics': self.config.enable_diagnostics,
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved complete training config to {config_path}")

    @staticmethod
    def load_agent(
        model_path: Path,
        agent_type: str,
        env: Optional[Any] = None
    ):
        """
        Load a trained agent from disk.

        This is a wrapper around model_utils.load_rl_agent() for compatibility.

        Args:
            model_path: Path to saved model
            agent_type: Type of agent (e.g., 'ppo', 'recurrent_ppo', 'sac', 'qrdqn')
            env: Optional environment

        Returns:
            Loaded agent (stable-baselines3 model)
        """
        from .model_utils import load_rl_agent

        # Use the centralized model loading function
        return load_rl_agent(Path(model_path), env=env)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced RL Trading Training")
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., GOOGL)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)',
                       default=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'))
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)',
                       default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo', 'recurrent_ppo', 'sac', 'qrdqn'])

    # Improvement flags
    parser.add_argument('--use-improvements', action='store_true',
                       help='Enable all improvements (recommended)')
    parser.add_argument('--no-action-masking', action='store_true',
                       help='Disable action masking')
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning')
    parser.add_argument('--use-6-actions', action='store_true',
                       help='Use improved 6-action space')

    parser.add_argument('--save-dir', type=str, help='Save directory')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    # Build configuration
    use_improvements = args.use_improvements

    config = EnhancedTrainingConfig(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        agent_type=args.agent,
        total_timesteps=args.timesteps,
        use_action_masking=use_improvements and not args.no_action_masking,
        use_enhanced_rewards=use_improvements,
        use_adaptive_sizing=use_improvements,
        use_improved_actions=args.use_6_actions,
        use_curriculum_learning=use_improvements and not args.no_curriculum,
        enable_diagnostics=True,
        save_dir=args.save_dir,
        verbose=args.verbose
    )

    # Train
    trainer = EnhancedRLTrainer(config)
    results = trainer.train()

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
