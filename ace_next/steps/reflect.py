"""ReflectStep — analyses a trace to produce a ReflectorOutput."""

from __future__ import annotations

import logging

from ..context import ACEStepContext
from ..outputs import AgentOutput
from ..protocols import ReflectorLike

logger = logging.getLogger(__name__)


class ReflectStep:
    """Run the Reflector role against the trace and current skillbook.

    Receives ``ctx.trace`` — a dict from EvaluateStep (standard ACE pipeline),
    a raw object from TraceAnalyser, or any integration-produced trace.  When
    the trace is a dict with known keys, the step extracts them and calls the
    Reflector's existing API.  For raw/opaque traces, it passes them as
    keyword arguments for the Reflector to handle.

    Declares ``async_boundary = True`` — everything from this step onward
    runs in a background thread pool when the pipeline has background
    execution enabled.

    Pure — produces a reflection object, no side effects.
    """

    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflection"})

    async_boundary = True
    max_workers = 3

    def __init__(self, reflector: ReflectorLike) -> None:
        self.reflector = reflector

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace = ctx.trace

        if isinstance(trace, dict):
            # Structured trace from EvaluateStep — extract known fields
            agent_output = AgentOutput(
                reasoning=trace.get("reasoning", ""),
                final_answer=trace.get("answer", ""),
                skill_ids=trace.get("skill_ids", []),
            )
            reflection = self.reflector.reflect(
                question=trace.get("question", ""),
                agent_output=agent_output,
                skillbook=ctx.skillbook,
                ground_truth=trace.get("ground_truth"),
                feedback=trace.get("feedback"),
            )
        else:
            # Raw trace from TraceAnalyser or integration — pass as-is
            # The Reflector must handle the trace type via **kwargs
            reflection = self.reflector.reflect(
                question="",
                agent_output=AgentOutput(reasoning="", final_answer=""),
                skillbook=ctx.skillbook,
                trace=trace,
            )

        return ctx.replace(reflection=reflection)
