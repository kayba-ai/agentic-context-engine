"""Reflector — analyzes agent outputs to extract lessons and improve strategies.

Uses PydanticAI for structured output validation with automatic retry
and error feedback.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.settings import ModelSettings

from ..core.context import SkillbookView
from ..core.outputs import AgentOutput, ReflectorOutput
from ..core.skillbook import Skillbook
from ..providers.pydantic_ai import resolve_model
from .helpers import extract_cited_skill_ids, format_optional, make_skillbook_excerpt
from .prompts import REFLECTOR_PROMPT, REFLECTOR_SKILL_EVAL_SECTION

logger = logging.getLogger(__name__)


class Reflector:
    """Analyzes agent outputs to extract lessons and improve strategies.

    The Reflector is the second ACE role. It analyzes the Agent's output
    and environment feedback to understand what went right or wrong,
    classifying which skillbook skills were helpful, harmful, or neutral.

    This implementation supports **SIMPLE** mode only (single-pass
    reflection). Recursive mode is handled by :mod:`ace.steps.rr`.

    Args:
        model: Model identifier string. Supports any LiteLLM model
            or PydanticAI-native identifier.
        prompt_template: Custom prompt template (defaults to
            :data:`REFLECTOR_PROMPT`).
        max_retries: Maximum retries for structured output validation.
        model_settings: Optional PydanticAI ``ModelSettings``.

    Example::

        reflector = Reflector("gpt-4o-mini")
        reflection = reflector.reflect(
            question="What is 2+2?",
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth="4",
            feedback="Correct!",
        )
        print(reflection.key_insight)
    """

    def __init__(
        self,
        model: str,
        *,
        prompt_template: str = REFLECTOR_PROMPT,
        max_retries: int = 3,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self._prompt_template = prompt_template
        self._agent = PydanticAgent(
            resolve_model(model),
            output_type=ReflectorOutput,
            retries=max_retries,
            model_settings=model_settings,
            defer_model_check=True,
        )

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Union[SkillbookView, Skillbook],
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        mode: str = "online",
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Analyze agent performance and extract learnings.

        This method signature matches :class:`ReflectorLike`.

        Args:
            question: The original question.
            agent_output: The agent's output to analyze.
            skillbook: Current skillbook (needs ``get_skill``).
            ground_truth: Expected correct answer (if available).
            feedback: Environment feedback text.
            mode: ``"online"`` includes skill evaluation and tagging;
                ``"offline"`` skips it.
            **kwargs: Accepted for protocol compatibility but not forwarded.

        Returns:
            :class:`ReflectorOutput` with analysis and skill tags.
        """
        skillbook_excerpt = make_skillbook_excerpt(skillbook, agent_output.skill_ids)

        if skillbook_excerpt:
            skillbook_context = f"Strategies Applied:\n{skillbook_excerpt}"
        else:
            skillbook_context = "(No strategies cited - outcome-based learning)"

        prompt = self._prompt_template.format(
            question=question,
            reasoning=agent_output.reasoning,
            prediction=agent_output.final_answer,
            ground_truth=format_optional(ground_truth),
            feedback=format_optional(feedback),
            skillbook_excerpt=skillbook_context,
        )

        # In online mode, append skill evaluation instructions
        if mode == "online" and skillbook_excerpt:
            prompt += "\n\n" + REFLECTOR_SKILL_EVAL_SECTION

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
