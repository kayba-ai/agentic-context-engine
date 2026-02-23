"""EvaluateStep — bridges the execute head to the learning tail."""

from __future__ import annotations

from ..context import ACEStepContext
from ..environments import TaskEnvironment


class EvaluateStep:
    """Bundle agent output into a trace dict, optionally evaluating with an environment.

    Always produces a ``trace`` dict with the structured fields from the
    execute head (question, context, ground_truth, reasoning, answer,
    skill_ids).  When an environment is provided, its feedback is included.

    The environment is injected at construction time — not on the context —
    to keep the context free of per-runner dependencies.
    """

    requires = frozenset({"sample", "agent_output"})
    provides = frozenset({"trace"})

    def __init__(self, environment: TaskEnvironment | None = None) -> None:
        self.environment = environment

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace: dict = {
            "question": ctx.sample.question,
            "context": ctx.sample.context,
            "ground_truth": ctx.sample.ground_truth,
            "reasoning": ctx.agent_output.reasoning,
            "answer": ctx.agent_output.final_answer,
            "skill_ids": ctx.agent_output.skill_ids,
        }
        if self.environment:
            result = self.environment.evaluate(
                sample=ctx.sample,
                agent_output=ctx.agent_output,
            )
            trace["feedback"] = result.feedback
        return ctx.replace(trace=trace)
