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
from .runners import ACE, ACERunner, TraceAnalyser

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
    # Runners
    "ACE",
    "ACERunner",
    "TraceAnalyser",
]
