"""SkillManager — transforms reflections into actionable skillbook updates.

Uses PydanticAI for structured output validation with automatic retry
and error feedback.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Union

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.settings import ModelSettings

from ..core.context import SkillbookView
from ..core.outputs import ReflectorOutput, SkillManagerOutput
from ..core.skillbook import Skillbook
from ..providers.pydantic_ai import resolve_model
from .prompts import SKILL_MANAGER_PROMPT

logger = logging.getLogger(__name__)


class SkillManager:
    """Transforms reflections into actionable skillbook updates.

    The SkillManager is the third ACE role. It analyzes the Reflector's
    output and decides how to update the skillbook — adding new
    strategies, updating existing ones, or removing harmful patterns.

    .. note::

        In ``ace``, deduplication is handled by a separate
        :class:`DeduplicateStep` in the pipeline. The SkillManager
        role only produces :class:`SkillManagerOutput`; it does not call
        a dedup manager itself.

    Args:
        model: Model identifier string. Supports any LiteLLM model
            or PydanticAI-native identifier.
        prompt_template: Custom prompt template (defaults to
            :data:`SKILL_MANAGER_PROMPT`).
        max_retries: Maximum retries for structured output validation.
        model_settings: Optional PydanticAI ``ModelSettings``.

    Example::

        sm = SkillManager("gpt-4o-mini")
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
        model: str,
        *,
        prompt_template: str = SKILL_MANAGER_PROMPT,
        max_retries: int = 3,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self._prompt_template = prompt_template
        self._agent = PydanticAgent(
            resolve_model(model),
            output_type=SkillManagerOutput,
            retries=max_retries,
            model_settings=model_settings,
            defer_model_check=True,
        )

    def update_skills(
        self,
        *,
        reflections: tuple[ReflectorOutput, ...],
        skillbook: Union[SkillbookView, Skillbook],
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """Generate update operations based on the reflections.

        This method signature matches :class:`SkillManagerLike`.

        Args:
            reflections: Tuple of Reflector analyses (1-tuple for single,
                N-tuple for batch).
            skillbook: Current skillbook (needs ``as_prompt``, ``stats``).
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
            }
            for r in reflections
        ]

        prompt = self._prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflections=json.dumps(reflections_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        result = self._agent.run_sync(prompt)
        output = result.output
        usage = result.usage()
        output.raw = {
            "usage": {
                "prompt_tokens": usage.input_tokens or 0,
                "completion_tokens": usage.output_tokens or 0,
                "total_tokens": usage.total_tokens or 0,
            },
        }
        return output
