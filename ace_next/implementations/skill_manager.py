"""SkillManager — transforms reflections into actionable skillbook updates."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..core.outputs import ReflectorOutput, SkillManagerOutput
from ..protocols.llm import LLMClientLike
from .prompts import SKILL_MANAGER_PROMPT

logger = logging.getLogger(__name__)


class SkillManager:
    """Transforms reflections into actionable skillbook updates.

    The SkillManager is the third ACE role. It analyzes the Reflector's
    output and decides how to update the skillbook — adding new
    strategies, updating existing ones, or removing harmful patterns.

    .. note::

        In ``ace_next``, deduplication is handled by a separate
        :class:`DeduplicateStep` in the pipeline. The SkillManager
        role only produces :class:`SkillManagerOutput`; it does not call
        a dedup manager itself.

    Args:
        llm: An LLM client that satisfies :class:`LLMClientLike`.
        prompt_template: Custom prompt template (defaults to
            :data:`SKILL_MANAGER_PROMPT`).

    Example::

        sm = SkillManager(llm)
        output = sm.update_skills(
            reflections=(reflection_output,),
            skillbook=skillbook,
            question_context="Math problem solving",
            progress="5/10 correct",
        )
        skillbook.apply_update(output.update)
    """

    def __init__(
        self,
        llm: LLMClientLike,
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
        reflections: tuple[ReflectorOutput, ...],
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """Generate update operations based on the reflections.

        This method signature matches :class:`SkillManagerLike`.

        Args:
            reflections: Tuple of Reflector analyses (1-tuple for single,
                N-tuple for batch).
            skillbook: Current skillbook (duck-typed, needs
                ``as_prompt``, ``stats``).
            question_context: Description of the task domain.
            progress: Current progress summary (e.g. ``"5/10 correct"``).
            **kwargs: Accepted for protocol compatibility but not forwarded.

        Returns:
            :class:`SkillManagerOutput` containing the update operations.
        """
        reflections_data = [
            {
                "reasoning": r.reasoning,
                "error_identification": r.error_identification,
                "root_cause_analysis": r.root_cause_analysis,
                "correct_approach": r.correct_approach,
                "key_insight": r.key_insight,
                "extracted_learnings": [
                    l.model_dump() for l in r.extracted_learnings
                ],
            }
            for r in reflections
        ]

        prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflections=json.dumps(reflections_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        return self.llm.complete_structured(
            prompt, SkillManagerOutput, max_retries=self.max_retries
        )
