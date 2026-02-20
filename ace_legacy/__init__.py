"""Agentic Context Engineering (ACE) — minimal core."""

from .skillbook import Skill, Skillbook
from .updates import UpdateOperation, UpdateBatch
from .llm import LLMClient, LLMResponse, DummyLLMClient, TransformersLLMClient
from .roles import (
    Agent,
    ReplayAgent,
    Reflector,
    SkillManager,
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
)
from .environments import (
    Sample,
    TaskEnvironment,
    SimpleEnvironment,
    EnvironmentResult,
    ACEStepResult,
)
from .adaptation import OfflineACE, OnlineACE, ACEBase
from .features import has_opik, has_litellm
from .prompt_manager import PromptManager

# Optional: LiteLLMClient (requires litellm)
LiteLLMClient = None
if has_litellm():
    try:
        from .llm_providers import LiteLLMClient as _LiteLLMClient

        LiteLLMClient = _LiteLLMClient  # type: ignore[assignment]
    except ImportError:
        pass

__all__ = [
    # Core data
    "Skill",
    "Skillbook",
    "UpdateOperation",
    "UpdateBatch",
    # LLM interface
    "LLMClient",
    "LLMResponse",
    "DummyLLMClient",
    "TransformersLLMClient",
    "LiteLLMClient",
    # Roles
    "Agent",
    "ReplayAgent",
    "Reflector",
    "SkillManager",
    "AgentOutput",
    "ReflectorOutput",
    "SkillManagerOutput",
    # Environments
    "Sample",
    "TaskEnvironment",
    "SimpleEnvironment",
    "EnvironmentResult",
    "ACEStepResult",
    # Adaptation loops
    "OfflineACE",
    "OnlineACE",
    "ACEBase",
    # Utilities
    "PromptManager",
    "has_opik",
    "has_litellm",
]
