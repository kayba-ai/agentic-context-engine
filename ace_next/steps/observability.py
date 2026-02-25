"""ObservabilityStep — logs pipeline metrics to Opik."""

from __future__ import annotations

import logging

from ..core.context import ACEStepContext

logger = logging.getLogger(__name__)


class ObservabilityStep:
    """Log pipeline metrics to the observability backend.

    Optional side-effect step — only requires ``skillbook`` (always present).
    Reads other context fields optionally so the same step works in both
    ACE and TraceAnalyser pipelines.
    """

    requires = frozenset({"skillbook"})
    provides = frozenset()

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        metrics: dict = {"skill_count": len(ctx.skillbook)}

        if ctx.reflection:
            metrics["key_insight"] = ctx.reflection.key_insight
            metrics["learnings_count"] = len(ctx.reflection.extracted_learnings)
        if ctx.skill_manager_output:
            metrics["operations_count"] = len(ctx.skill_manager_output.operations)
        if ctx.trace:
            metrics["trace_type"] = type(ctx.trace).__name__

        logger.info("ObservabilityStep: %s", metrics)
        return ctx
