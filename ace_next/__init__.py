"""ACE next — pipeline-based rewrite of the ACE framework."""

# Core types
from .core import (
    EnvironmentResult,
    Sample,
    SimpleEnvironment,
    Skill,
    Skillbook,
    TaskEnvironment,
    UpdateBatch,
    UpdateOperation,
)
from .deduplication import DeduplicationManager, SimilarityDetector
from .implementations import Agent, Reflector, SkillManager
from .protocols import DeduplicationConfig
from .integrations import wrap_skillbook_context
from .providers import (
    InstructorClient,
    LiteLLMClient,
    wrap_with_instructor,
)
from .runners import (
    ACE,
    ACELiteLLM,
    BrowserUse,
    ClaudeCode,
    LangChain,
    TraceAnalyser,
)
from .steps.opik import OPIK_AVAILABLE, OpikStep, register_opik_litellm_callback

__all__ = [
    # Core data types
    "Skill",
    "Skillbook",
    "UpdateOperation",
    "UpdateBatch",
    # Environments
    "Sample",
    "EnvironmentResult",
    "TaskEnvironment",
    "SimpleEnvironment",
    # Implementations
    "Agent",
    "Reflector",
    "SkillManager",
    # LLM providers
    "LiteLLMClient",
    "InstructorClient",
    "wrap_with_instructor",
    # Runners
    "ACE",
    "ACELiteLLM",
    "BrowserUse",
    "ClaudeCode",
    "LangChain",
    "TraceAnalyser",
    # Deduplication
    "DeduplicationConfig",
    "DeduplicationManager",
    "SimilarityDetector",
    # Observability
    "OpikStep",
    "OPIK_AVAILABLE",
    "register_opik_litellm_callback",
    # Utilities
    "wrap_skillbook_context",
]
