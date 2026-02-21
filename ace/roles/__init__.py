"""ACE roles — Agent, Reflector, and SkillManager components.

All public symbols are re-exported here so that existing imports like
``from ace.roles import Agent`` continue to work unchanged.
"""

from .agent import Agent, AgentOutput, ReplayAgent
from .reflector import (
    ExtractedLearning,
    Reflector,
    ReflectorMode,
    ReflectorOutput,
    SkillTag,
)
from .skill_manager import SkillManager, SkillManagerOutput
from ._helpers import extract_cited_skill_ids, _safe_json_loads

__all__ = [
    # Agent
    "Agent",
    "AgentOutput",
    "ReplayAgent",
    # Reflector
    "ExtractedLearning",
    "Reflector",
    "ReflectorMode",
    "ReflectorOutput",
    "SkillTag",
    # SkillManager
    "SkillManager",
    "SkillManagerOutput",
    # Helpers (public)
    "extract_cited_skill_ids",
]
