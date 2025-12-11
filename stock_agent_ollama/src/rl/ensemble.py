"""
Ensemble Agent - Combines PPO and RecurrentPPO

This module implements an ensemble trading agent that combines the strengths of:
- PPO: Aggressive growth strategy
- RecurrentPPO: Risk-managed strategy with LSTM memory

The ensemble uses a weighted voting mechanism where both agents contribute to
the final decision based on their respective strengths.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnsemblePPOAgent:
    """
    Ensemble agent combining PPO and RecurrentPPO.

    **Strategy**:
    - PPO (30%): Aggressive growth for opportunistic trades
    - RecurrentPPO (70%): Primary strategy with LSTM memory and trend-following

    **Decision Logic**:
    1. Get predictions from both agents
    2. If agents agree: Use that action
    3. If agents disagree: Use weighted vote (PPO 30%, RecurrentPPO 70%)
    4. Confidence-based weighting: Higher confidence gets more weight

    **Rationale for 30/70 weighting**:
    - RecurrentPPO shows superior risk-adjusted returns (better Sharpe ratios)
    - LSTM memory handles volatility better than standard PPO
    - 70% weight prevents "sell amplification" when both agents vote to sell
    - 30% PPO weight still provides growth focus when conditions are right
    """

    def __init__(
        self,
        ppo_model,
        recurrent_ppo_model,
        ppo_weight: float = 0.3,
        recurrent_ppo_weight: float = 0.7,
        use_confidence: bool = True
    ):
        """
        Initialize ensemble agent.

        Args:
            ppo_model: Trained PPO model
            recurrent_ppo_model: Trained RecurrentPPO model
            ppo_weight: Weight for PPO decisions (0-1)
            recurrent_ppo_weight: Weight for RecurrentPPO decisions (0-1)
            use_confidence: Whether to use action probabilities for weighting
        """
        self.ppo = ppo_model
        self.recurrent_ppo = recurrent_ppo_model
        self.ppo_weight = ppo_weight
        self.recurrent_ppo_weight = recurrent_ppo_weight
        self.use_confidence = use_confidence

        # Normalize weights
        total_weight = ppo_weight + recurrent_ppo_weight
        self.ppo_weight /= total_weight
        self.recurrent_ppo_weight /= total_weight

        # Statistics
        self.agreement_count = 0
        self.disagreement_count = 0
        self.predictions_count = 0

        logger.info(f"Ensemble created: PPO={self.ppo_weight:.1%}, RecurrentPPO={self.recurrent_ppo_weight:.1%}")

    @property
    def observation_space(self):
        """
        Get observation space.
        Returns the observation space of the component with the most features (RecurrentPPO),
        which is the superset of required features.
        """
        return self.recurrent_ppo.observation_space

    def _adjust_observation_for_ppo(self, observation: np.ndarray) -> np.ndarray:
        """
        Adjust observation for PPO model if it expects fewer features (e.g. no trend indicators or masks).
        """
        if not hasattr(self.ppo, 'observation_space'):
            return observation
            
        ppo_features = self.ppo.observation_space.shape[-1]
        input_features = observation.shape[-1]
        
        if input_features == ppo_features:
            return observation
            
        # Case 1: Input has Trend(3) but PPO doesn't want it (32 -> 29)
        if input_features == ppo_features + 3:
            try:
                if observation.ndim > 1:
                    return np.concatenate([observation[..., :10], observation[..., 13:]], axis=-1)
                else:
                    return np.concatenate([observation[:10], observation[13:]])
            except Exception as e:
                logger.warning(f"Error slicing observation for PPO (Case 1): {e}")

        # Case 2: Input has Trend(3) + Mask(6) but PPO wants neither (32 -> 23)
        if input_features == 32 and ppo_features == 23:
            try:
                if observation.ndim > 1:
                    return np.concatenate([observation[..., :10], observation[..., 13:26]], axis=-1)
                else:
                    return np.concatenate([observation[:10], observation[13:26]])
            except Exception as e:
                logger.warning(f"Error slicing observation for PPO (Case 2): {e}")

        # Case 3: Input has Mask(6) but PPO doesn't want it (29 -> 23 or 32 -> 26 if Trend matched)
        if input_features == ppo_features + 6:
            try:
                if observation.ndim > 1:
                    return observation[..., :-6]
                else:
                    return observation[:-6]
            except Exception as e:
                logger.warning(f"Error slicing observation for PPO (Case 3): {e}")

        logger.warning(f"PPO Obs shape mismatch: Input {observation.shape}, Expects {self.ppo.observation_space.shape}. Using full.")
        return observation

    def _adjust_observation_for_recurrent_ppo(self, observation: np.ndarray) -> np.ndarray:
        """
        Adjust observation for RecurrentPPO model if it expects fewer features (e.g. no masks).
        """
        if not hasattr(self.recurrent_ppo, 'observation_space'):
            return observation
            
        rppo_features = self.recurrent_ppo.observation_space.shape[-1]
        input_features = observation.shape[-1]
        
        if input_features == rppo_features:
            return observation
            
        # Case 1: Input has Mask(6) but RPPO doesn't want it (e.g., 32 -> 26)
        if input_features == rppo_features + 6:
            try:
                if observation.ndim > 1:
                    return observation[..., :-6]
                else:
                    return observation[:-6]
            except Exception as e:
                logger.warning(f"Error slicing observation for RPPO (Case 1): {e}")

        logger.warning(f"RPPO Obs shape mismatch: Input {observation.shape}, Expects {self.recurrent_ppo.observation_space.shape}. Using full.")
        return observation

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Predict action using ensemble voting.

        Args:
            observation: Current environment observation
            state: RNN states (for RecurrentPPO)
            episode_start: Episode start flags
            deterministic: Whether to use deterministic prediction

        Returns:
            Tuple of (action, state)
        """
        self.predictions_count += 1

        # Adjust observations for each model's expected input space
        rppo_obs = self._adjust_observation_for_recurrent_ppo(observation)
        ppo_obs = self._adjust_observation_for_ppo(observation)

        # Get PPO prediction
        ppo_action, _ = self.ppo.predict(
            ppo_obs,
            deterministic=deterministic
        )

        # Get RecurrentPPO prediction
        rppo_action, new_state = self.recurrent_ppo.predict(
            rppo_obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic
        )

        # Extract scalar actions
        if isinstance(ppo_action, np.ndarray):
            ppo_action_val = int(ppo_action.item() if ppo_action.ndim == 0 else ppo_action[0])
        else:
            ppo_action_val = int(ppo_action)

        if isinstance(rppo_action, np.ndarray):
            rppo_action_val = int(rppo_action.item() if rppo_action.ndim == 0 else rppo_action[0])
        else:
            rppo_action_val = int(rppo_action)

        # Check agreement
        if ppo_action_val == rppo_action_val:
            self.agreement_count += 1
            final_action = ppo_action_val
            logger.debug(f"Agreement: Both chose action {final_action}")
        else:
            self.disagreement_count += 1

            if self.use_confidence:
                # Get action probabilities (pass appropriate observations)
                final_action = self._weighted_vote_with_confidence(
                    ppo_obs, rppo_obs, ppo_action_val, rppo_action_val, state, episode_start
                )
            else:
                # Simple weighted vote based on fixed weights
                final_action = self._weighted_vote(ppo_action_val, rppo_action_val)

            logger.debug(f"Disagreement: PPO={ppo_action_val}, RecurrentPPO={rppo_action_val}, Final={final_action}")

        # Return action in same format as input
        if isinstance(ppo_action, np.ndarray):
            final_action = np.array([final_action])

        return final_action, new_state

    def _weighted_vote(self, ppo_action: int, rppo_action: int) -> int:
        """Simple weighted vote using fixed weights."""
        # Random selection based on weights
        if np.random.random() < self.ppo_weight:
            return ppo_action
        else:
            return rppo_action

    def _weighted_vote_with_confidence(
        self,
        ppo_observation: np.ndarray,
        rppo_observation: np.ndarray,
        ppo_action: int,
        rppo_action: int,
        state: Optional[Tuple],
        episode_start: Optional[np.ndarray]
    ) -> int:
        """Weighted vote using action probabilities as confidence."""
        
        # Conflict Resolution: Default to HOLD (0) if actions are opposing (Buy vs Sell)
        # This prevents forced gambling when models disagree on direction.
        # Assumes ImprovedTradingAction space: HOLD=0, BUY={1,2,3}, SELL={4,5}
        is_ppo_buy = ppo_action in [1, 2, 3]
        is_ppo_sell = ppo_action in [4, 5]
        is_rppo_buy = rppo_action in [1, 2, 3]
        is_rppo_sell = rppo_action in [4, 5]
        
        if (is_ppo_buy and is_rppo_sell) or (is_ppo_sell and is_rppo_buy):
            logger.debug(f"Conflict Resolution: Opposing actions (PPO={ppo_action}, RPPO={rppo_action}) -> HOLD")
            return 0  # HOLD
            
        try:
            # Get action probabilities from both models (with appropriate observations)
            ppo_probs = self._get_action_probabilities(self.ppo, ppo_observation, None)
            rppo_probs = self._get_action_probabilities(
                self.recurrent_ppo, rppo_observation, state, episode_start
            )

            # Get confidence for each agent's chosen action
            ppo_confidence = ppo_probs[ppo_action]
            rppo_confidence = rppo_probs[rppo_action]

            # Weight by base weight * confidence
            ppo_score = self.ppo_weight * ppo_confidence
            rppo_score = self.recurrent_ppo_weight * rppo_confidence

            # Choose action with higher weighted confidence
            if ppo_score > rppo_score:
                logger.debug(f"PPO wins: {ppo_score:.3f} vs {rppo_score:.3f}")
                return ppo_action
            else:
                logger.debug(f"RecurrentPPO wins: {rppo_score:.3f} vs {ppo_score:.3f}")
                return rppo_action

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}, using simple weighted vote")
            return self._weighted_vote(ppo_action, rppo_action)

    def _get_action_probabilities(
        self,
        model,
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get action probability distribution from model."""
        from stable_baselines3.common.utils import obs_as_tensor
        import torch

        # Ensure batch dimension exists
        # If observation matches the single observation space, add batch dim
        if observation.shape == model.observation_space.shape:
            observation = observation.reshape(1, *observation.shape)
        
        # Convert to tensor
        obs_tensor = obs_as_tensor(observation, model.device)

        # For RecurrentPPO
        if hasattr(model.policy, 'lstm_actor'):
            if state is None:
                # Initialize zero state if not provided
                if hasattr(model.policy, 'lstm_hidden_state_shape'):
                    shape = model.policy.lstm_hidden_state_shape
                    # LSTM needs tuple of (hidden, cell)
                    hidden = torch.zeros(shape).to(model.device)
                    cell = torch.zeros(shape).to(model.device)
                    state = (hidden, cell)
                else:
                    # Fallback if shape attribute missing (shouldn't happen in newer sb3-contrib)
                    # Assuming standard (1, 1, 256)
                    hidden = torch.zeros((1, 1, 256)).to(model.device)
                    cell = torch.zeros((1, 1, 256)).to(model.device)
                    state = (hidden, cell)
            
            # Check if state is numpy (from predict return) and convert to tensor
            elif isinstance(state, tuple) and isinstance(state[0], np.ndarray):
                hidden = torch.as_tensor(state[0]).to(model.device)
                cell = torch.as_tensor(state[1]).to(model.device)
                state = (hidden, cell)

            
            if episode_start is None:
                episode_start = np.array([False])

            # Convert episode_start to tensor (float) as expected by RecurrentPPO policy
            episode_start_tensor = torch.as_tensor(episode_start).float().to(model.device)

            dist_result = model.policy.get_distribution(obs_tensor, state, episode_start_tensor)
            
            # Handle tuple return (distribution, new_states) if applicable
            if isinstance(dist_result, tuple):
                distribution = dist_result[0]
            else:
                distribution = dist_result
        else:
            # For regular PPO
            distribution = model.policy.get_distribution(obs_tensor)

        # Get probabilities
        probs = distribution.distribution.probs.detach().cpu().numpy()[0]
        return probs

    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        total = self.agreement_count + self.disagreement_count
        agreement_rate = self.agreement_count / total if total > 0 else 0

        return {
            'predictions': self.predictions_count,
            'agreements': self.agreement_count,
            'disagreements': self.disagreement_count,
            'agreement_rate': agreement_rate,
            'ppo_weight': self.ppo_weight,
            'recurrent_ppo_weight': self.recurrent_ppo_weight,
        }

    def save(self, save_dir: Path):
        """Save ensemble metadata."""
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save ensemble configuration
        import json
        config = {
            'ensemble_type': 'PPO_RecurrentPPO',
            'ppo_weight': float(self.ppo_weight),
            'recurrent_ppo_weight': float(self.recurrent_ppo_weight),
            'use_confidence': self.use_confidence,
            'statistics': self.get_statistics()
        }

        with open(save_dir / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Ensemble config saved to {save_dir}/ensemble_config.json")