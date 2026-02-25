"""Agent — produces answers using the current skillbook of strategies."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.outputs import AgentOutput
from ..protocols.llm import LLMClientLike
from .helpers import extract_cited_skill_ids, format_optional
from .prompts import AGENT_PROMPT

logger = logging.getLogger(__name__)


class Agent:
    """Produces answers using the current skillbook of strategies.

    The Agent is one of three core ACE roles. It takes a question and
    uses the accumulated strategies in the skillbook to produce reasoned
    answers.

    Args:
        llm: An LLM client that satisfies :class:`LLMClientLike`.
        prompt_template: Custom prompt template (defaults to
            :data:`AGENT_PROMPT`).
        max_retries: Maximum validation retries (forwarded to the LLM
            client if it supports it).

    Example::

        agent = Agent(llm)
        output = agent.generate(
            question="What is the capital of France?",
            context="Answer concisely",
            skillbook=skillbook,
        )
        print(output.final_answer)  # "Paris"
    """

    def __init__(
        self,
        llm: LLMClientLike,
        prompt_template: str = AGENT_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        skillbook: Any,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Generate an answer using skillbook strategies.

        This method signature matches :class:`AgentLike`.

        Args:
            question: The question to answer.
            context: Additional context or requirements.
            skillbook: Current skillbook (duck-typed, needs ``as_prompt``).
            reflection: Optional reflection from a previous attempt.
            **kwargs: Forwarded to the LLM client.

        Returns:
            :class:`AgentOutput` with reasoning, final_answer, and
            cited skill_ids.
        """
        prompt = self.prompt_template.format(
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            reflection=format_optional(reflection),
            question=question,
            context=format_optional(context),
        )

        output: AgentOutput = self.llm.complete_structured(
            prompt, AgentOutput, **kwargs
        )
        output.skill_ids = extract_cited_skill_ids(output.reasoning)
        return output
