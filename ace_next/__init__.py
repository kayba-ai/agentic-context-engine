"""ACE next — pipeline-based rewrite of the ACE framework."""

# Core types
from .context import ACESample, ACEStepContext, SkillbookView
from .environments import EnvironmentResult, Sample, SimpleEnvironment, TaskEnvironment
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
    ReflectorLike,
    SkillManagerLike,
)
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
    "ReflectorLike",
    "SkillManagerLike",
    "DeduplicationConfig",
    "DeduplicationManagerLike",
]
