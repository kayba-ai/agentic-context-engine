"""Reflector role — analyzes agent outputs to extract lessons and improve strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..llm import LLMClient
from ..skillbook import Skillbook
from ..prompt_manager import PromptManager
from .agent import AgentOutput
from ._helpers import (
    _format_optional,
    _make_skillbook_excerpt,
    _maybe_wrap_with_instructor,
    maybe_track,
)

# Default prompt (v2.1 with {current_date} filled in)
_prompt_manager = PromptManager(default_version="2.1")
REFLECTOR_PROMPT = _prompt_manager.get_reflector_prompt()


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

class Reflector:
    """
    Analyzes agent outputs to extract lessons and improve strategies.

    The Reflector is the second ACE role. It analyzes the Agent's output
    and environment feedback to understand what went right or wrong, classifying
    which skillbook skills were helpful, harmful, or neutral.

    Args:
        llm: The LLM client to use for reflection
        prompt_template: Custom prompt template (uses REFLECTOR_PROMPT by default)
        max_retries: Maximum validation retries via Instructor (default: 3)

    Example:
        >>> from ace import Reflector, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> reflector = Reflector(client)
        >>>
        >>> reflection = reflector.reflect(
        ...     question="What is 2+2?",
        ...     agent_output=agent_output,
        ...     skillbook=skillbook,
        ...     ground_truth="4",
        ...     feedback="Correct!"
        ... )
        >>> print(reflection.key_insight)
        Successfully solved the arithmetic problem
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = _maybe_wrap_with_instructor(llm, max_retries)
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    @maybe_track(
        name="reflector_reflect",
        tags=["ace-framework", "role", "reflector"],
        project_name="ace-roles",
    )
    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Skillbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        return self._reflect_impl(
            question=question,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    def _reflect_impl(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Skillbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        **kwargs: Any,
    ) -> ReflectorOutput:
        skillbook_excerpt = _make_skillbook_excerpt(skillbook, agent_output.skill_ids)

        # Format skillbook section based on citation presence
        if skillbook_excerpt:
            skillbook_context = f"Strategies Applied:\n{skillbook_excerpt}"
        else:
            skillbook_context = "(No strategies cited - outcome-based learning)"

        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=agent_output.reasoning,
            prediction=agent_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            skillbook_excerpt=skillbook_context,
        )

        # Filter out non-LLM kwargs (like 'sample' used for ReplayAgent)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        # Use Instructor for automatic validation (always available - core dependency)
        return self.llm.complete_structured(base_prompt, ReflectorOutput, **llm_kwargs)
