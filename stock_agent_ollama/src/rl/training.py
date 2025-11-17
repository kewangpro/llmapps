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
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

from .environments import EnhancedTradingEnv
from .improvements import (
    EnhancedRewardConfig,
    PPORewardConfig,
    CurriculumManager,
    TrainingDiagnostics
)
from .callbacks import TrainingProgressCallback, PerformanceMonitorCallback

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingConfig:
    """Configuration for enhanced RL training."""

    # Basic settings
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = 100000.0

    # Environment enhancements
    use_action_masking: bool = True  # Always enabled for safety
    use_enhanced_rewards: bool = True
    use_adaptive_sizing: bool = True
    use_improved_actions: bool = True  # Always enabled - 6-action space with HOLD as default
    # Disabled curriculum learning - was causing excessive exploration leading to 65% invalid action rate
    # Agent performs better with direct learning on the full task
    use_curriculum_learning: bool = False
    enable_diagnostics: bool = True
    use_lstm_policy: bool = False # New field to enable LSTM policy

    # Position limits
    max_position_size: int = 1000
    max_position_pct: float = 40.0

    # Reward configuration
    reward_config: EnhancedRewardConfig = field(default_factory=EnhancedRewardConfig)

    # Agent settings
    agent_type: str = "ppo"
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
    total_timesteps: int = 100000  # Reduced from 300k to focus on quality learning with lower invalid action rate
    eval_freq: int = 5000
    save_freq: int = 10000

    # Transaction costs
    # RESTORED to 0.0005 for DQN compatibility
    # DQN works best with light transaction costs (learns optimal trading frequency)
    # PPO/A2C get higher costs via PPORewardConfig (0.002)
    transaction_cost_rate: float = 0.0005  # 0.05% per trade (DQN-optimized)
    slippage_rate: float = 0.0005

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
        # DQN uses EnhancedRewardConfig (lighter penalties, proven to work)
        # PPO/A2C use PPORewardConfig (stronger penalties to fight action collapse)
        if config.reward_config is None or isinstance(config.reward_config, EnhancedRewardConfig):
            if config.agent_type.lower() in ['ppo', 'a2c']:
                logger.info(f"Using PPORewardConfig for {config.agent_type.upper()}")
                self.config.reward_config = PPORewardConfig()
            else:  # DQN or other
                logger.info(f"Using EnhancedRewardConfig (DQN-optimized) for {config.agent_type.upper()}")
                if config.reward_config is None:
                    self.config.reward_config = EnhancedRewardConfig()

        # Setup save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Add LSTM prefix for RecurrentPPO models
            if config.use_lstm_policy and config.agent_type.lower() == 'ppo':
                model_name = f"lstm_ppo_{config.symbol}_{timestamp}"
            else:
                model_name = f"{config.agent_type}_{config.symbol}_{timestamp}"
            self.save_dir = Path(f"data/models/rl/{model_name}")

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def setup_environment(self) -> EnhancedTradingEnv:
        """Create enhanced trading environment."""

        # Setup curriculum learning
        if self.config.use_curriculum_learning:
            self.curriculum_manager = CurriculumManager()
            logger.info(f"Curriculum learning enabled. Starting at stage: {self.curriculum_manager.get_current_stage().name}")
        else:
            self.curriculum_manager = None

        # Create environment
        env = EnhancedTradingEnv(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_balance=self.config.initial_balance,
            transaction_cost_rate=self.config.transaction_cost_rate,
            slippage_rate=self.config.slippage_rate,
            max_position_size=self.config.max_position_size,
            max_position_pct=self.config.max_position_pct,
            use_action_masking=self.config.use_action_masking,
            use_enhanced_rewards=self.config.use_enhanced_rewards,
            use_adaptive_sizing=self.config.use_adaptive_sizing,
            use_improved_actions=self.config.use_improved_actions,
            reward_config=self.config.reward_config,
            curriculum_manager=self.curriculum_manager,
            enable_diagnostics=self.config.enable_diagnostics
        )

        self.env = env
        return env

    def setup_agent(self) -> Any:
        """Create RL agent."""
        if self.env is None:
            self.setup_environment()

        # Common agent parameters
        agent_params = {
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'verbose': self.config.verbose,
        }

        # Determine agent class and policy type
        agent_class = None
        policy_type = 'MlpPolicy' # Default policy

        if self.config.use_lstm_policy:
            if self.config.agent_type.lower() == 'ppo':
                agent_class = RecurrentPPO
                policy_type = 'MlpLstmPolicy'
            elif self.config.agent_type.lower() == 'a2c':
                # RecurrentA2C does not exist, fall back to standard A2C
                logger.warning(f"LSTM policy is not supported for A2C in sb3_contrib. Falling back to MlpPolicy for A2C.")
                agent_class = A2C
            else: # DQN or unsupported agent type
                logger.warning(f"LSTM policy is not supported for {self.config.agent_type.upper()}. Falling back to MlpPolicy.")
                if self.config.agent_type.lower() == 'dqn':
                    agent_class = DQN
                else:
                    raise ValueError(f"Unsupported agent type: {self.config.agent_type}. Available: ppo, a2c, dqn")
        else: # Not using LSTM policy
            if self.config.agent_type.lower() == 'ppo':
                agent_class = PPO
            elif self.config.agent_type.lower() == 'a2c':
                agent_class = A2C
            elif self.config.agent_type.lower() == 'dqn':
                agent_class = DQN
            else:
                raise ValueError(f"Unsupported agent type: {self.config.agent_type}. Available: ppo, a2c, dqn")

        # PPO-specific parameters
        if self.config.agent_type.lower() == 'ppo':
            agent_params.update({
                'n_steps': self.config.n_steps,
                'batch_size': self.config.batch_size,
                'n_epochs': self.config.n_epochs,
                'ent_coef': self.config.ent_coef,
            })
        elif self.config.agent_type.lower() == 'a2c':
            # A2C-specific parameters to prevent action collapse and improve stability
            agent_params.update({
                'ent_coef': self.config.ent_coef,
                'n_steps': 128,              # Increase from default 5 to reduce variance
                'gae_lambda': 0.95,          # Reduce from default 1.0 for better bias-variance tradeoff
                'normalize_advantage': True, # Enable advantage normalization for stability
                'vf_coef': 0.5,             # Value function coefficient
                'max_grad_norm': 0.5,       # Gradient clipping
            })
        elif self.config.agent_type.lower() == 'dqn':
            # DQN-specific parameters optimized for stock trading
            agent_params.update({
                'buffer_size': 100000,           # Large replay buffer for experience reuse
                'learning_starts': 10000,        # Fill buffer before learning
                'batch_size': self.config.batch_size,
                'tau': 0.005,                    # Soft target network update
                'train_freq': 4,                 # Update every 4 steps
                'gradient_steps': 1,
                'target_update_interval': 1000,  # Hard target update every 1000 steps
                'exploration_fraction': 0.3,     # 30% of training for exploration
                'exploration_initial_eps': 1.0,  # Start fully random
                'exploration_final_eps': 0.05,   # End with 5% random actions
            })

        self.agent = agent_class(
            policy_type,
            self.env,
            **agent_params
        )

        logger.info(f"Created {self.config.agent_type.upper()} agent with {policy_type}")
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

        # Diagnostics
        if self.config.enable_diagnostics:
            self.callbacks.append(DiagnosticsCallback(
                env=self.env,
                report_freq=10000,
                verbose=self.config.verbose
            ))

    def train(self) -> Dict[str, Any]:
        """Execute enhanced training."""

        if self.env is None:
            self.setup_environment()

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

        # Save configuration
        self.save_config()

        # Get diagnostics
        diagnostics = self.env.get_diagnostics_summary() if self.config.enable_diagnostics else {}

        # Get training stats from progress callback
        progress_callback = None
        for callback in self.callbacks:
            if isinstance(callback, TrainingProgressCallback):
                progress_callback = callback
                break

        training_stats = progress_callback.get_training_stats() if progress_callback else {}

        # Extract explained variance from model logger (PPO/A2C specific)
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
            }
        }

        if self.config.verbose > 0:
            self._print_training_summary(results)

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
        """Save training configuration."""
        config_path = self.save_dir / "training_config.json"

        config_dict = {
            'symbol': self.config.symbol,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'initial_balance': self.config.initial_balance,
            'use_action_masking': self.config.use_action_masking,
            'use_enhanced_rewards': self.config.use_enhanced_rewards,
            'use_adaptive_sizing': self.config.use_adaptive_sizing,
            'use_improved_actions': self.config.use_improved_actions,
            'use_curriculum_learning': self.config.use_curriculum_learning,
            'use_lstm_policy': self.config.use_lstm_policy,
            'agent_type': self.config.agent_type,
            'learning_rate': self.config.learning_rate,
            'total_timesteps': self.config.total_timesteps,
            'max_position_pct': self.config.max_position_pct,
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved configuration to {config_path}")

    @staticmethod
    def load_agent(
        model_path: Path,
        agent_type: str,
        env: Optional[Any] = None
    ):
        """
        Load a trained agent from disk.

        Args:
            model_path: Path to saved model
            agent_type: Type of agent (e.g., 'ppo', 'a2c', 'dqn')
            env: Optional environment (not used, kept for compatibility)

        Returns:
            Loaded agent (stable-baselines3 model)
        """
        from stable_baselines3 import PPO, A2C, DQN
        from sb3_contrib import RecurrentPPO

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Check if this is an LSTM model by checking the training config
        model_dir = model_path.parent
        config_path = model_dir / "training_config.json"
        use_lstm = False

        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                use_lstm = config.get('use_lstm_policy', False)

        # Load using appropriate class based on agent type and LSTM flag
        if agent_type.lower() == 'ppo':
            if use_lstm:
                logger.info(f"Loading RecurrentPPO model from {model_path}")
                agent = RecurrentPPO.load(str(model_path))
            else:
                agent = PPO.load(str(model_path))
        elif agent_type.lower() == 'a2c':
            # A2C doesn't have recurrent version, always use standard
            if use_lstm:
                logger.warning("Loaded A2C model was trained with LSTM requested, but A2C doesn't support LSTM")
            agent = A2C.load(str(model_path))
        elif agent_type.lower() == 'dqn':
            # DQN doesn't have recurrent version, always use standard
            if use_lstm:
                logger.warning("Loaded DQN model was trained with LSTM requested, but DQN doesn't support LSTM")
            agent = DQN.load(str(model_path))
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}. Available: ppo, a2c, dqn")

        # Add is_trained attribute for compatibility with backtest code
        agent.is_trained = True

        return agent


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
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo', 'a2c'])

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
