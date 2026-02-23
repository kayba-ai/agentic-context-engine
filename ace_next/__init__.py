"""ACE next — pipeline-based rewrite of the ACE framework."""

# Core types
from .context import ACESample, ACEStepContext, SkillbookView
from .deduplication import DeduplicationManager, SimilarityDetector
from .environments import EnvironmentResult, Sample, SimpleEnvironment, TaskEnvironment
from .implementations import Agent, Reflector, SkillManager
from .outputs import (
    AgentOutput,
    ExtractedLearning,
    ReflectorOutput,
    SkillManagerOutput,
    SkillTag,
)
from .protocols import (
    AgentLike,
    DeduplicationConfig,
    DeduplicationManagerLike,
    LLMClientLike,
    ReflectorLike,
    SkillManagerLike,
)
from .runners import ACE, ACERunner, TraceAnalyser
from .skill import Skill, SimilarityDecision
from .skillbook import Skillbook
from .updates import UpdateBatch, UpdateOperation

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
