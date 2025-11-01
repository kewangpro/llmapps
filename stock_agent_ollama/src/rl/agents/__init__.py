"""RL agents for trading strategies."""

from .base_agent import BaseRLAgent
from .ppo_agent import PPOAgent
from .a2c_agent import A2CAgent

__all__ = [
    'BaseRLAgent',
    'PPOAgent',
    'A2CAgent',
    'create_agent',
]


def create_agent(
    agent_type: str,
    env,
    **kwargs
) -> BaseRLAgent:
    """
    Factory function to create RL agents.

    Args:
        agent_type: Type of agent ('ppo', 'a2c')
        env: Trading environment
        **kwargs: Additional agent parameters

    Returns:
        Instantiated agent

    Raises:
        ValueError: If agent_type is unknown
    """
    agent_type = agent_type.lower()

    if agent_type == 'ppo':
        return PPOAgent(env, **kwargs)
    elif agent_type == 'a2c':
        return A2CAgent(env, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: ppo, a2c")
