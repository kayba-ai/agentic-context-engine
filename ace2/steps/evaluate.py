"""EvaluateStep — runs the task environment to score the agent's output."""

from __future__ import annotations

import logging

from pipeline import StepContext

logger = logging.getLogger(__name__)


class EvaluateStep:
    """Pipeline step that evaluates the agent's output against the environment.

    Calls ``environment.evaluate(sample, agent_output)`` and writes
    ``environment_result`` back into the context.
    """

    requires = frozenset({"sample", "agent_output", "environment"})
    provides = frozenset({"environment_result"})

    def __call__(self, ctx: StepContext) -> StepContext:
        environment_result = ctx.environment.evaluate(ctx.sample, ctx.agent_output)
        return ctx.replace(environment_result=environment_result)
