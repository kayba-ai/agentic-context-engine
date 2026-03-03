"""RROpikStep -- log Recursive Reflector traces to Opik.

Pure side-effect step that reads ``ctx.reflection.raw["rr_trace"]``
(populated by :class:`RRStep`) and creates a hierarchical Opik trace
with child spans per REPL iteration.

Place after ``RRStep`` in the pipeline.  Gracefully degrades to a
no-op when Opik is not installed or is disabled.

**Explicit opt-in only** -- constructing an ``RROpikStep`` is the
opt-in signal.  Opik is never auto-enabled just because the package
is installed.

Resulting trace hierarchy::

    rr_reflect (trace)
    +-- rr_iteration_0 (span)
    +-- rr_iteration_1 (span)
    +-- rr_iteration_2 (span)   <-- FINAL called here

Each iteration span logs: code sent to sandbox, stdout/stderr,
terminated flag.  Sub-agent call history is attached to the parent
trace metadata.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ace_next.core.context import ACEStepContext

logger = logging.getLogger(__name__)

# Soft-import Opik -- RROpikStep is a no-op when the package is absent.
try:
    import opik as _opik

    OPIK_AVAILABLE = True
except ImportError:
    _opik = None  # type: ignore[assignment]
    OPIK_AVAILABLE = False


def _opik_disabled() -> bool:
    """Check environment variables for Opik disable signals."""
    if os.environ.get("OPIK_DISABLED", "").lower() in ("true", "1", "yes"):
        return True
    if os.environ.get("OPIK_ENABLED", "").lower() in ("false", "0", "no"):
        return True
    return False


class RROpikStep:
    """Log Recursive Reflector REPL traces to Opik.

    Pure side-effect step -- reads ``ctx.reflection.raw["rr_trace"]``
    and creates one Opik trace per RR invocation with child spans per
    iteration.  Never mutates the context.

    Args:
        project_name: Opik project name.
        tags: Tags applied to every trace.
    """

    requires = frozenset({"reflection"})
    provides = frozenset()

    def __init__(
        self,
        project_name: str = "ace-rr",
        tags: list[str] | None = None,
    ) -> None:
        self.project_name = project_name
        self.tags = tags or ["ace", "rr"]
        self._client: Any | None = None
        self.enabled = OPIK_AVAILABLE and not _opik_disabled()

        if self.enabled:
            try:
                api_key = os.environ.get("OPIK_API_KEY")
                workspace = os.environ.get("OPIK_WORKSPACE")
                host = os.environ.get("OPIK_URL_OVERRIDE")
                self._client = _opik.Opik(
                    project_name=project_name,
                    api_key=api_key or None,
                    workspace=workspace or None,
                    host=host or None,
                )
            except Exception as exc:
                logger.debug("RROpikStep: failed to create Opik client: %s", exc)
                self.enabled = False

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if not self.enabled:
            return ctx

        if not ctx.reflection:
            return ctx

        rr_trace = ctx.reflection.raw.get("rr_trace")
        if not rr_trace:
            return ctx

        try:
            self._log_trace(ctx, rr_trace)
        except Exception as exc:
            logger.debug("RROpikStep: failed to log trace (non-critical): %s", exc)

        return ctx

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_trace(self, ctx: ACEStepContext, rr_trace: dict[str, Any]) -> None:
        """Build and send an Opik trace with child spans from RR data."""
        iterations = rr_trace.get("iterations", [])
        subagent_calls = rr_trace.get("subagent_calls", [])

        trace_input = self._build_input(ctx)
        trace_output = self._build_output(ctx, rr_trace)
        metadata = self._build_metadata(rr_trace, subagent_calls)
        tags = list(self.tags)

        trace = self._client.trace(
            name="rr_reflect",
            input=trace_input,
            output=trace_output,
            metadata=metadata,
            tags=tags,
            project_name=self.project_name,
        )

        # Create a child span per iteration
        for entry in iterations:
            span_input = {"code": entry.get("code")}
            span_output = {
                "stdout": entry.get("stdout"),
                "stderr": entry.get("stderr"),
            }
            span_metadata = {
                "iteration": entry.get("iteration"),
                "terminated": entry.get("terminated", False),
            }
            span = trace.span(
                name=f"rr_iteration_{entry.get('iteration', '?')}",
                input=span_input,
                output=span_output,
                metadata=span_metadata,
            )
            span.end()

        trace.end()

    def _build_input(self, ctx: ACEStepContext) -> dict[str, Any]:
        """Extract input data from the context."""
        result: dict[str, Any] = {}
        if isinstance(ctx.trace, dict):
            result["question"] = ctx.trace.get("question", "")
            if ctx.trace.get("ground_truth"):
                result["ground_truth"] = ctx.trace["ground_truth"]
            if ctx.trace.get("feedback"):
                result["feedback"] = ctx.trace["feedback"]
        return result

    def _build_output(
        self, ctx: ACEStepContext, rr_trace: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract output data from the reflection."""
        result: dict[str, Any] = {}
        if ctx.reflection:
            result["key_insight"] = ctx.reflection.key_insight
            result["reasoning"] = ctx.reflection.reasoning[:500]
            result["learnings_count"] = len(ctx.reflection.extracted_learnings)
        if rr_trace.get("timed_out"):
            result["timed_out"] = True
        return result

    def _build_metadata(
        self,
        rr_trace: dict[str, Any],
        subagent_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build metadata dict for the parent trace."""
        metadata: dict[str, Any] = {
            "total_iterations": rr_trace.get("total_iterations", 0),
        }
        if subagent_calls:
            metadata["subagent_call_count"] = len(subagent_calls)
            metadata["subagent_calls"] = subagent_calls
        return metadata

    def flush(self) -> None:
        """Drain buffered traces before process exit."""
        if self._client is not None and hasattr(self._client, "flush"):
            try:
                self._client.flush()
            except Exception:
                pass
