"""Reflector role — analyzes agent outputs to extract lessons and improve strategies."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    from ..reflector.config import RecursiveConfig

logger = logging.getLogger(__name__)

# Default prompt (v2.1 with {current_date} filled in)
_prompt_manager = PromptManager(default_version="2.1")
REFLECTOR_PROMPT = _prompt_manager.get_reflector_prompt()


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------


class ReflectorMode(Enum):
    """Mode of operation for the Reflector.

    Attributes:
        SIMPLE: Single-pass reflection (default, 1 LLM call)
        RECURSIVE: Code execution with REPL loop (3-15 LLM calls)
        AUTO: Automatically select based on trace complexity
    """

    SIMPLE = "simple"
    RECURSIVE = "recursive"
    AUTO = "auto"


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

    Supports two modes:
        - SIMPLE (default): Single-pass LLM reflection
        - RECURSIVE: Code-execution REPL with trace analysis
        - AUTO: Selects based on reasoning length

    Args:
        llm: The LLM client to use for reflection
        prompt_template: Custom prompt template (uses REFLECTOR_PROMPT by default)
        mode: ReflectorMode (default: SIMPLE)
        recursive_config: RecursiveConfig for RECURSIVE mode
        max_retries: Maximum validation retries via Instructor (default: 3)
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        mode: ReflectorMode = ReflectorMode.SIMPLE,
        recursive_config: Optional["RecursiveConfig"] = None,
        max_retries: int = 3,
    ) -> None:
        self._raw_llm = llm
        self.llm = _maybe_wrap_with_instructor(llm, max_retries)
        self.prompt_template = prompt_template
        self.mode = mode
        self.recursive_config = recursive_config
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
        selected = self._select_mode(agent_output)

        if selected == ReflectorMode.RECURSIVE:
            return self._recursive_reflect(
                question=question,
                agent_output=agent_output,
                skillbook=skillbook,
                ground_truth=ground_truth,
                feedback=feedback,
                **kwargs,
            )

        return self._reflect_impl(
            question=question,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    def _select_mode(self, agent_output: AgentOutput) -> ReflectorMode:
        """Select reflection mode based on configuration and trace complexity."""
        if self.mode == ReflectorMode.AUTO:
            reasoning_tokens = len(agent_output.reasoning) // 4
            if reasoning_tokens > 2000:
                logger.debug(
                    f"AUTO mode: selecting RECURSIVE (~{reasoning_tokens} tokens)"
                )
                return ReflectorMode.RECURSIVE
            logger.debug(f"AUTO mode: selecting SIMPLE (~{reasoning_tokens} tokens)")
            return ReflectorMode.SIMPLE
        return self.mode

    def _recursive_reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Skillbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Perform recursive reflection using code execution."""
        from ..reflector import RecursiveReflector

        traces: Optional[List[Dict[str, Any]]] = kwargs.pop("traces", None)

        if traces is None and agent_output is not None:
            # Legacy path: convert agent_output to traces
            traces = [
                {
                    "role": "agent",
                    "reasoning": agent_output.reasoning,
                    "answer": agent_output.final_answer,
                    "skill_ids": agent_output.skill_ids,
                }
            ]

        recursive_reflector = RecursiveReflector(
            llm=self._raw_llm,
            config=self.recursive_config,
        )

        return recursive_reflector.reflect(
            question=question,
            traces=traces or [],
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            agent_output=agent_output,
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

        # Filter out non-LLM kwargs (like 'sample', 'traces' used for ReplayAgent / RR)
        llm_kwargs = {k: v for k, v in kwargs.items() if k not in ("sample", "traces")}

        # Use Instructor for automatic validation (always available - core dependency)
        return self.llm.complete_structured(base_prompt, ReflectorOutput, **llm_kwargs)
