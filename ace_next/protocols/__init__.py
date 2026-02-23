"""Public contracts — protocols that steps depend on, not concrete classes."""

from .agent import AgentLike
from .deduplication import DeduplicationConfig, DeduplicationManagerLike
from .reflector import ReflectorLike
from .skill_manager import SkillManagerLike

__all__ = [
    "AgentLike",
    "DeduplicationConfig",
    "DeduplicationManagerLike",
    "ReflectorLike",
    "SkillManagerLike",
]
