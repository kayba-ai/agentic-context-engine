"""Agentic Context Engineering (ACE) reproduction framework."""

from typing import Optional
from .skillbook import Skill, Skillbook
from .updates import UpdateOperation, UpdateBatch
from .llm import LLMClient, DummyLLMClient, TransformersLLMClient
from .roles import (
    Agent,
    ReplayAgent,
    Reflector,
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
    ACEStepResult,
)
from .async_learning import (
    LearningTask,
    ReflectionResult,
    ThreadSafeSkillbook,
    AsyncLearningPipeline,
)

# Import optional feature detection
from .features import has_opik, has_litellm

# Import observability components if available
if has_opik():
    try:
        from .observability import OpikIntegration as _OpikIntegration

        OpikIntegration: Optional[type] = _OpikIntegration
        OBSERVABILITY_AVAILABLE = True
    except ImportError:
        OpikIntegration: Optional[type] = None  # type: ignore
        OBSERVABILITY_AVAILABLE = False
else:
    OpikIntegration: Optional[type] = None  # type: ignore
    OBSERVABILITY_AVAILABLE = False

# Import production LLM clients if available
if has_litellm():
    try:
        from .llm_providers import LiteLLMClient as _LiteLLMClient

        LiteLLMClient: Optional[type] = _LiteLLMClient
        LITELLM_AVAILABLE = True
    except ImportError:
        LiteLLMClient: Optional[type] = None  # type: ignore
        LITELLM_AVAILABLE = False
else:
    LiteLLMClient: Optional[type] = None  # type: ignore
    LITELLM_AVAILABLE = False

# Import integrations (LiteLLM, browser-use, LangChain, Claude Code, etc.) if available
try:
    from .integrations import (
        ACELiteLLM as _ACELiteLLM,
        ACEAgent as _ACEAgent,
        ACELangChain as _ACELangChain,
        ACEClaudeCode as _ACEClaudeCode,
        wrap_skillbook_context as _wrap_skillbook_context,
        BROWSER_USE_AVAILABLE as _BROWSER_USE_AVAILABLE,
        LANGCHAIN_AVAILABLE as _LANGCHAIN_AVAILABLE,
        CLAUDE_CODE_AVAILABLE as _CLAUDE_CODE_AVAILABLE,
    )

    ACELiteLLM: Optional[type] = _ACELiteLLM
    ACEAgent: Optional[type] = _ACEAgent
    ACELangChain: Optional[type] = _ACELangChain
    ACEClaudeCode: Optional[type] = _ACEClaudeCode
    wrap_skillbook_context: Optional[type] = _wrap_skillbook_context  # type: ignore
    BROWSER_USE_AVAILABLE = _BROWSER_USE_AVAILABLE
    LANGCHAIN_AVAILABLE = _LANGCHAIN_AVAILABLE
    CLAUDE_CODE_AVAILABLE = _CLAUDE_CODE_AVAILABLE
except ImportError:
    ACELiteLLM: Optional[type] = None  # type: ignore
    ACEAgent: Optional[type] = None  # type: ignore
    ACELangChain: Optional[type] = None  # type: ignore
    ACEClaudeCode: Optional[type] = None  # type: ignore
    wrap_skillbook_context: Optional[type] = None  # type: ignore
    BROWSER_USE_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False
    CLAUDE_CODE_AVAILABLE = False

# Import deduplication module
from .deduplication import (
    DeduplicationConfig,
    DeduplicationManager,
)

__all__ = [
    # Core components
    "Skill",
    "Skillbook",
    "UpdateOperation",
    "UpdateBatch",
    "LLMClient",
    "DummyLLMClient",
    "TransformersLLMClient",
    "LiteLLMClient",
    "Agent",
    "ReplayAgent",
    "Reflector",
    "SkillManager",
    "AgentOutput",
    "ReflectorOutput",
    "SkillManagerOutput",
    "OfflineACE",
    "OnlineACE",
    "ACEBase",
    "Sample",
    "TaskEnvironment",
    "SimpleEnvironment",
    "EnvironmentResult",
    "ACEStepResult",
    # Deduplication
    "DeduplicationConfig",
    "DeduplicationManager",
    # Out-of-box integrations
    "ACELiteLLM",  # LiteLLM integration (quick start)
    "ACEAgent",  # Browser-use integration
    "ACELangChain",  # LangChain integration (complex workflows)
    "ACEClaudeCode",  # Claude Code CLI integration
    # Utilities
    "wrap_skillbook_context",
    # Async learning
    "LearningTask",
    "ReflectionResult",
    "ThreadSafeSkillbook",
    "AsyncLearningPipeline",
    # Feature flags
    "OpikIntegration",
    "LITELLM_AVAILABLE",
    "OBSERVABILITY_AVAILABLE",
    "BROWSER_USE_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
    "CLAUDE_CODE_AVAILABLE",
]
