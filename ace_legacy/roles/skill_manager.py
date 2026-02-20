"""SkillManager role — transforms reflections into actionable skillbook updates."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..llm import LLMClient
from ..skillbook import Skillbook
from ..updates import UpdateBatch
from ..prompt_manager import PromptManager
from .reflector import ReflectorOutput

logger = logging.getLogger(__name__)

# Default prompt (v2.1 with {current_date} filled in)
_prompt_manager = PromptManager(default_version="2.1")
SKILL_MANAGER_PROMPT = _prompt_manager.get_skill_manager_prompt()


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class SkillManagerOutput(BaseModel):
    """Output from the SkillManager role containing skillbook update operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    update: UpdateBatch = Field(
        ..., description="Batch of update operations to apply to skillbook"
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_flat_response(cls, data: Any) -> Any:
        """Accept both nested {"update": {...}} and flat {"reasoning": ..., "operations": [...]}."""
        if isinstance(data, dict) and "update" not in data and "operations" in data:
            data = {"update": data}
        return data


# ---------------------------------------------------------------------------
# SkillManager
# ---------------------------------------------------------------------------

class SkillManager:
    """
    Transforms reflections into actionable skillbook updates.

    The SkillManager is the third ACE role. It analyzes the Reflector's output
    and decides how to update the skillbook - adding new strategies, updating
    existing ones, or removing harmful patterns.

    Args:
        llm: The LLM client to use for skill management
        prompt_template: Custom prompt template (uses SKILL_MANAGER_PROMPT by default)
        max_retries: Maximum validation retries (default: 3)

    The SkillManager emits UpdateOperations:
        - ADD: Add new strategy skills
        - UPDATE: Modify existing skills
        - TAG: Update helpful/harmful counts
        - REMOVE: Delete unhelpful skills
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = SKILL_MANAGER_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def update_skills(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Skillbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        return self._update_skills_impl(
            reflection=reflection,
            skillbook=skillbook,
            question_context=question_context,
            progress=progress,
            **kwargs,
        )

    def _update_skills_impl(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Skillbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """
        Generate update operations to modify the skillbook based on reflection.

        Args:
            reflection: The Reflector's analysis of what went right/wrong
            skillbook: Current skillbook to potentially update
            question_context: Description of the task domain or question type
            progress: Current progress summary (e.g., "5/10 correct")
            **kwargs: Additional arguments passed to the LLM

        Returns:
            SkillManagerOutput containing the update operations to apply
        """
        # Serialize reflection with all meaningful fields
        reflection_data = {
            "reasoning": reflection.reasoning,
            "error_identification": reflection.error_identification,
            "root_cause_analysis": reflection.root_cause_analysis,
            "correct_approach": reflection.correct_approach,
            "key_insight": reflection.key_insight,
            "extracted_learnings": [
                l.model_dump() for l in reflection.extracted_learnings
            ],
        }

        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflection=json.dumps(reflection_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        # Filter out non-LLM kwargs (like 'sample' used for ReplayAgent)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        return self.llm.complete_structured(
            base_prompt, SkillManagerOutput, **llm_kwargs
        )
