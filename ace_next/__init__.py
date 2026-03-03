"""ACE next — pipeline-based rewrite of the ACE framework."""

# Pipeline engine (re-exported from pipeline/)
from pipeline import Branch, MergeStrategy, Pipeline, SampleResult, StepProtocol

# ACE context
from .core import ACEStepContext, SkillbookView

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

# Runners
from .runners import (
    ACE,
    ACELiteLLM,
    ACERunner,
    BrowserUse,
    ClaudeCode,
    LangChain,
    TraceAnalyser,
)
from .rr import RRConfig, RRStep

# Steps
from .steps import (
    AgentStep,
    ApplyStep,
    CheckpointStep,
    DeduplicateStep,
    EvaluateStep,
    ExportSkillbookMarkdownStep,
    LoadTracesStep,
    ObservabilityStep,
    PersistStep,
    ReflectStep,
    TagStep,
    UpdateStep,
    learning_tail,
)
from .steps.opik import OPIK_AVAILABLE, OpikStep, register_opik_litellm_callback

__all__ = [
    # Pipeline composition
    "Pipeline",
    "Branch",
    "MergeStrategy",
    "StepProtocol",
    "SampleResult",
    # ACE context
    "ACEStepContext",
    "SkillbookView",
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
    "ACERunner",
    "BrowserUse",
    "ClaudeCode",
    "LangChain",
    "TraceAnalyser",
    # Steps
    "AgentStep",
    "EvaluateStep",
    "ReflectStep",
    "TagStep",
    "UpdateStep",
    "ApplyStep",
    "DeduplicateStep",
    "CheckpointStep",
    "LoadTracesStep",
    "ExportSkillbookMarkdownStep",
    "ObservabilityStep",
    "PersistStep",
    "learning_tail",
    # Recursive Reflector
    "RRStep",
    "RRConfig",
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
