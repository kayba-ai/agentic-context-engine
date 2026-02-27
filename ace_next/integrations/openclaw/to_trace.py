"""OpenClawToTraceStep — convert raw JSONL events to a structured trace dict."""

from __future__ import annotations

from ...core.context import ACEStepContext


class OpenClawToTraceStep:
    """Convert raw OpenClaw JSONL events into a structured trace dict.

    This step receives ``ctx.trace`` as a ``list[dict]`` of raw JSONL events
    (placed by ``LoadTracesStep``) and converts them into the trace dict
    format expected by ``ReflectStep``.

    Currently a **pass-through** — returns the context unchanged.
    Transformation logic will be defined separately.
    """

    requires = frozenset({"trace"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        # Pass-through: transformation logic to be defined later.
        return ctx
