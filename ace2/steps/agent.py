"""AgentStep — generates an answer using the current skillbook."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pipeline import StepContext

if TYPE_CHECKING:
    from ace.roles import Agent

logger = logging.getLogger(__name__)


class AgentStep:
    """Pipeline step that wraps the ACE Agent role.

    Reads the current sample, skillbook, and recent reflections from context,
    calls ``agent.generate()``, and writes ``agent_output`` back.
    """

    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"agent_output"})

    def __init__(self, agent: "Agent") -> None:
        self.agent = agent

    def __call__(self, ctx: StepContext) -> StepContext:
        # Build the reflection context string from the rolling window
        reflection_str = "\n---\n".join(ctx.recent_reflections) if ctx.recent_reflections else ""

        agent_output = self.agent.generate(
            question=ctx.sample.question,
            context=ctx.sample.context,
            skillbook=ctx.skillbook,
            reflection=reflection_str,
            sample=ctx.sample,  # Pass through for ReplayAgent support
        )

        return ctx.replace(agent_output=agent_output)
