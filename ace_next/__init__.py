"""ACE next — pipeline-based rewrite of the ACE framework."""

# Core types
from .core import (
    ACESample,
    ACEStepContext,
    AgentOutput,
    EnvironmentResult,
    ExtractedLearning,
    ReflectorOutput,
    Sample,
    SimpleEnvironment,
    Skill,
    Skillbook,
    SkillbookView,
    SkillManagerOutput,
    SkillTag,
    SimilarityDecision,
    TaskEnvironment,
    UpdateBatch,
    UpdateOperation,
)
from .deduplication import DeduplicationManager, SimilarityDetector
from .implementations import Agent, Reflector, SkillManager
from .protocols import (
    AgentLike,
    DeduplicationConfig,
    DeduplicationManagerLike,
    LLMClientLike,
    ReflectorLike,
    SkillManagerLike,
)
from .integrations import (
    BrowserExecuteStep,
    BrowserResult,
    BrowserToTrace,
    ClaudeCodeExecuteStep,
    ClaudeCodeResult,
    ClaudeCodeToTrace,
    LangChainExecuteStep,
    LangChainResult,
    LangChainToTrace,
    wrap_skillbook_context,
)
from .providers import (
    CLAUDE_CODE_CLI_AVAILABLE,
    ClaudeCodeLLMClient,
    ClaudeCodeLLMConfig,
    InstructorClient,
    LangChainLiteLLMClient,
    LiteLLMClient,
    LiteLLMConfig,
    LLMResponse,
    wrap_with_instructor,
)
from .runners import (
    ACE,
    ACELiteLLM,
    ACERunner,
    BrowserUse,
    ClaudeCode,
    LangChain,
    TraceAnalyser,
)
from .steps.opik import OPIK_AVAILABLE, OpikStep, register_opik_litellm_callback

__all__ = [
    # Context
    "ACESample",
    "ACEStepContext",
    "SkillbookView",
    # Data types
    "Skill",
    "SimilarityDecision",
    "Skillbook",
    "UpdateOperation",
    "UpdateBatch",
    # Outputs
    "AgentOutput",
    "ExtractedLearning",
    "ReflectorOutput",
    "SkillManagerOutput",
    "SkillTag",
    # Environments
    "Sample",
    "EnvironmentResult",
    "TaskEnvironment",
    "SimpleEnvironment",
    # Protocols
    "AgentLike",
    "LLMClientLike",
    "ReflectorLike",
    "SkillManagerLike",
    "DeduplicationConfig",
    "DeduplicationManagerLike",
    # Implementations
    "Agent",
    "Reflector",
    "SkillManager",
    # Deduplication
    "DeduplicationManager",
    "SimilarityDetector",
    # Integration steps
    "BrowserExecuteStep",
    "BrowserResult",
    "BrowserToTrace",
    "ClaudeCodeExecuteStep",
    "ClaudeCodeResult",
    "ClaudeCodeToTrace",
    "LangChainExecuteStep",
    "LangChainResult",
    "LangChainToTrace",
    "wrap_skillbook_context",
    # LLM providers
    "LiteLLMClient",
    "LiteLLMConfig",
    "LLMResponse",
    "InstructorClient",
    "wrap_with_instructor",
    "LangChainLiteLLMClient",
    "ClaudeCodeLLMClient",
    "ClaudeCodeLLMConfig",
    "CLAUDE_CODE_CLI_AVAILABLE",
    # Runners
    "ACE",
    "ACERunner",
    "BrowserUse",
    "ClaudeCode",
    "LangChain",
    "TraceAnalyser",
    # Observability
    "OpikStep",
    "OPIK_AVAILABLE",
    "register_opik_litellm_callback",
    # Convenience wrapper
    "ACELiteLLM",
]
