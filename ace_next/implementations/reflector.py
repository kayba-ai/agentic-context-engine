"""Reflector — analyzes agent outputs to extract lessons and improve strategies."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.outputs import AgentOutput, ReflectorOutput
from ..protocols.llm import LLMClientLike
from .helpers import format_optional, make_skillbook_excerpt
from .prompts import REFLECTOR_PROMPT

logger = logging.getLogger(__name__)


class Reflector:
    """Analyzes agent outputs to extract lessons and improve strategies.

    The Reflector is the second ACE role. It analyzes the Agent's output
    and environment feedback to understand what went right or wrong,
    classifying which skillbook skills were helpful, harmful, or neutral.

    This implementation supports **SIMPLE** mode only (single-pass
    reflection). Recursive mode can be added later.

    Args:
        llm: An LLM client that satisfies :class:`LLMClientLike`.
        prompt_template: Custom prompt template (defaults to
            :data:`REFLECTOR_PROMPT`).
        max_retries: Maximum validation retries (forwarded to the LLM
            client if it supports it).

    Example::

        reflector = Reflector(llm)
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
        llm: LLMClientLike,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Analyze agent performance and extract learnings.

        This method signature matches :class:`ReflectorLike`.

        Args:
            question: The original question.
            agent_output: The agent's output to analyze.
            skillbook: Current skillbook (duck-typed, needs
                ``get_skill``).
            ground_truth: Expected correct answer (if available).
            feedback: Environment feedback text.
            **kwargs: Forwarded to the LLM client.

        Returns:
            :class:`ReflectorOutput` with analysis and skill tags.
        """
        skillbook_excerpt = make_skillbook_excerpt(
            skillbook, agent_output.skill_ids
        )

        if skillbook_excerpt:
            skillbook_context = f"Strategies Applied:\n{skillbook_excerpt}"
        else:
            skillbook_context = "(No strategies cited - outcome-based learning)"

        prompt = self.prompt_template.format(
            question=question,
            reasoning=agent_output.reasoning,
            prediction=agent_output.final_answer,
            ground_truth=format_optional(ground_truth),
            feedback=format_optional(feedback),
            skillbook_excerpt=skillbook_context,
        )

        return self.llm.complete_structured(
            prompt, ReflectorOutput, **kwargs
        )
