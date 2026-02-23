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
from .skill import Skill, SimilarityDecision
from .skillbook import Skillbook
from .updates import UpdateBatch, UpdateOperation

# Protocols and config — defined alongside the steps that use them
from .steps.agent import AgentLike
from .steps.deduplicate import DeduplicationConfig, DeduplicationManagerLike
from .steps.reflect import ReflectorLike
from .steps.update import SkillManagerLike

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
    # Role protocols (defined in their step files)
    "AgentLike",
    "ReflectorLike",
    "SkillManagerLike",
    # Deduplication (defined in deduplicate step)
    "DeduplicationConfig",
    "DeduplicationManagerLike",
]
