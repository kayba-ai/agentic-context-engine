"""Core data types for the ACE framework."""

from .context import ACESample, ACEStepContext, SkillbookView
from .environments import EnvironmentResult, Sample, SimpleEnvironment, TaskEnvironment
from .insight_source import (
    TRACE_IDENTITY_METADATA_KEY,
    InsightSource,
    TraceIdentity,
)
from .outputs import (
    AgentOutput,
    ExtractedLearning,
    ReflectorOutput,
    SkillManagerOutput,
)
from .skillbook import (
    OperationType,
    Skill,
    Skillbook,
    SimilarityDecision,
    UpdateBatch,
    UpdateOperation,
)

__all__ = [
    # Skillbook types
    "VALID_SKILL_TAGS",
    "OperationType",
    "Skill",
    "Skillbook",
    "SimilarityDecision",
    "UpdateBatch",
    "UpdateOperation",
    # Outputs
    "AgentOutput",
    "ExtractedLearning",
    "ReflectorOutput",
    "SkillManagerOutput",
    # Context
    "ACESample",
    "ACEStepContext",
    "SkillbookView",
    # Environments
    "EnvironmentResult",
    "Sample",
    "SimpleEnvironment",
    "TaskEnvironment",
    # Provenance
    "InsightSource",
    "TraceIdentity",
    "TRACE_IDENTITY_METADATA_KEY",
]
