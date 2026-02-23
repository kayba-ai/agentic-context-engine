"""Output types produced by ACE roles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .updates import UpdateBatch


class AgentOutput(BaseModel):
    """Output from the Agent role containing reasoning and answer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reasoning: str = Field(..., description="Step-by-step reasoning process")
    final_answer: str = Field(..., description="The final answer to the question")
    skill_ids: List[str] = Field(
        default_factory=list, description="IDs of strategies cited in reasoning"
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )
    trace_context: Optional[Any] = Field(
        default=None,
        exclude=True,
        description="Pre-built TraceContext from integration (bypasses auto-detection)",
    )


class ExtractedLearning(BaseModel):
    """A single learning extracted by the Reflector from task execution."""

    learning: str = Field(..., description="The extracted learning or insight")
    atomicity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How atomic/focused this learning is"
    )
    evidence: str = Field(
        default="", description="Evidence from execution supporting this learning"
    )
    justification: str = Field(
        default="",
        description="Why this learning was chosen: generalizable pattern, explicit preference, etc.",
    )


class SkillTag(BaseModel):
    """Classification tag for a skill strategy (helpful/harmful/neutral)."""

    id: str = Field(..., description="The skill ID being tagged")
    tag: str = Field(
        ..., description="Classification: 'helpful', 'harmful', or 'neutral'"
    )


class ReflectorOutput(BaseModel):
    """Output from the Reflector role containing analysis and skill classifications."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reasoning: str = Field(..., description="Overall reasoning about the outcome")
    error_identification: str = Field(
        default="", description="Description of what went wrong (if applicable)"
    )
    root_cause_analysis: str = Field(
        default="", description="Analysis of why errors occurred"
    )
    correct_approach: str = Field(
        ..., description="What the correct approach should be"
    )
    key_insight: str = Field(
        ..., description="The main lesson learned from this iteration"
    )
    extracted_learnings: List[ExtractedLearning] = Field(
        default_factory=list, description="Learnings extracted from task execution"
    )
    skill_tags: List[SkillTag] = Field(
        default_factory=list, description="Classifications of strategy effectiveness"
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )


class SkillManagerOutput(BaseModel):
    """Output from the SkillManager role containing skillbook update operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    update: UpdateBatch = Field(
        ..., description="Batch of update operations to apply to skillbook"
    )
    consolidation_operations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Operations to consolidate similar skills (MERGE, DELETE, KEEP, UPDATE)",
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )
