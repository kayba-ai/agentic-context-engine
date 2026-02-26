"""Agentic Context Engineering (ACE) reproduction framework."""

from typing import Optional
from .skillbook import Skill, Skillbook
from .updates import UpdateOperation, UpdateBatch
from .llm import LLMClient, DummyLLMClient
from .roles import (
    Agent,
    ReplayAgent,
    Reflector,
    ReflectorMode,
    SkillManager,
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
)
from .adaptation import (
    OfflineACE,
    OnlineACE,
    ACEBase,
    Sample,
    TaskEnvironment,
    SimpleEnvironment,
    EnvironmentResult,
)
from .deduplication import DeduplicationConfig

# Import optional feature detection
from .features import has_litellm

# Import production LLM client if available
if has_litellm():
    try:
        from .llm_providers import LiteLLMClient as _LiteLLMClient

        LiteLLMClient: Optional[type] = _LiteLLMClient
    except ImportError:
        LiteLLMClient: Optional[type] = None  # type: ignore
else:
    LiteLLMClient: Optional[type] = None  # type: ignore

# Import integrations if available
try:
    from .integrations import (
        ACELiteLLM as _ACELiteLLM,
        ACEAgent as _ACEAgent,
        ACELangChain as _ACELangChain,
        ACEClaudeCode as _ACEClaudeCode,
        wrap_skillbook_context as _wrap_skillbook_context,
    )

    ACELiteLLM: Optional[type] = _ACELiteLLM
    ACEAgent: Optional[type] = _ACEAgent
    ACELangChain: Optional[type] = _ACELangChain
    ACEClaudeCode: Optional[type] = _ACEClaudeCode
    wrap_skillbook_context: Optional[type] = _wrap_skillbook_context  # type: ignore
except ImportError:
    ACELiteLLM: Optional[type] = None  # type: ignore
    ACEAgent: Optional[type] = None  # type: ignore
    ACELangChain: Optional[type] = None  # type: ignore
    ACEClaudeCode: Optional[type] = None  # type: ignore
    wrap_skillbook_context: Optional[type] = None  # type: ignore

__all__ = [
    # Core data types
    "Skill",
    "Skillbook",
    "UpdateOperation",
    "UpdateBatch",
    # LLM clients
    "LLMClient",
    "DummyLLMClient",
    "LiteLLMClient",
    # Roles
    "Agent",
    "ReplayAgent",
    "Reflector",
    "ReflectorMode",
    "SkillManager",
    "AgentOutput",
    "ReflectorOutput",
    "SkillManagerOutput",
    # Adaptation loops
    "ACEBase",
    "OfflineACE",
    "OnlineACE",
    "Sample",
    "TaskEnvironment",
    "SimpleEnvironment",
    "EnvironmentResult",
    # Integrations
    "ACELiteLLM",
    "ACEAgent",
    "ACELangChain",
    "ACEClaudeCode",
    # Configuration
    "DeduplicationConfig",
    # Utilities
    "wrap_skillbook_context",
]
