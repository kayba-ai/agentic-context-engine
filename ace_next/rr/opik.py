"""RROpikStep -- log Recursive Reflector traces to Opik.

Pure side-effect step that iterates ``ctx.reflections`` and reads
``reflection.raw["rr_trace"]`` from each (populated by :class:`RRStep`)
to create a hierarchical Opik trace with child spans per REPL iteration
and sub-agent call.

Place after ``RRStep`` in the pipeline.  Gracefully degrades to a
no-op when Opik is not installed or is disabled.

**Explicit opt-in only** -- constructing an ``RROpikStep`` is the
opt-in signal.  Opik is never auto-enabled just because the package
is installed.

This step replaces the need for a separate LiteLLM Opik callback --
all LLM telemetry (model, tokens, cost) is captured from the
``LLMResponse.raw`` metadata that flows through the iteration log
and sub-agent call history.

Resulting trace hierarchy::

    rr_reflect (trace)
    +-- rr_iteration_0 (span)  — LLM call: model, tokens, cost
    +-- rr_iteration_1 (span)
    +-- rr_iteration_2 (span)  <-- FINAL called here
    +-- subagent_call_1 (span) — sub-agent LLM call details
    +-- subagent_call_2 (span)
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

    Pure side-effect step -- iterates ``ctx.reflections`` and reads
    ``reflection.raw["rr_trace"]`` from each to create one Opik trace
    per reflection with child spans per iteration and sub-agent call.
    Never mutates the context.

    All LLM telemetry (model, tokens, cost) is read from
    ``llm_metadata`` fields already captured in the iteration log and
    sub-agent call history -- no separate LiteLLM callback needed.

    Args:
        project_name: Opik project name.
        tags: Tags applied to every trace.
    """

    requires: frozenset[str] = frozenset({"reflections"})
    provides: frozenset[str] = frozenset()

    def __init__(
        self,
        project_name: str = "ace-rr",
        tags: list[str] | None = None,
        thread_id: str | None = None,
    ) -> None:
        self.project_name = project_name
        self.tags = tags or ["ace", "rr"]
        self.thread_id = thread_id
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
                logger.warning("RROpikStep: failed to create Opik client: %s", exc)
                self.enabled = False

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if not self.enabled:
            return ctx

        for reflection in ctx.reflections:
            rr_trace = reflection.raw.get("rr_trace")
            if not rr_trace:
                continue
            try:
                self._log_trace(ctx, reflection, rr_trace)
            except Exception as exc:
                logger.debug("RROpikStep: failed to log trace (non-critical): %s", exc)

        # Flush after each sample so traces appear in Opik in real time
        self.flush()

        return ctx

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_trace(
        self,
        ctx: ACEStepContext,
        reflection: Any,
        rr_trace: dict[str, Any],
    ) -> None:
        """Build and send an Opik trace with child spans from RR data."""
        iterations = rr_trace.get("iterations", [])
        subagent_calls = rr_trace.get("subagent_calls", [])

        trace_input = self._build_input(ctx)
        trace_output = self._build_output(reflection, rr_trace)
        metadata = self._build_metadata(rr_trace, iterations, subagent_calls)
        tags = list(self.tags)

        assert self._client is not None
        # Build informative trace name
        trace_name = "rr_reflect"
        if isinstance(ctx.trace, dict):
            q = ctx.trace.get("question", "")
            if q:
                trace_name = f"rr: {q[:80]}"
        trace_kwargs: dict[str, Any] = dict(
            name=trace_name,
            input=trace_input,
            output=trace_output,
            metadata=metadata,
            tags=tags,
            project_name=self.project_name,
        )
        if self.thread_id:
            trace_kwargs["thread_id"] = self.thread_id
        trace = self._client.trace(**trace_kwargs)

        # Child span per REPL iteration (includes LLM call telemetry)
        for entry in iterations:
            self._log_iteration_span(trace, entry)

        # Child span per sub-agent call
        for call in subagent_calls:
            self._log_subagent_span(trace, call)

        trace.end()

    def _log_iteration_span(
        self, trace: Any, entry: dict[str, Any]
    ) -> None:
        """Create a child span for a single REPL iteration."""
        llm_meta = entry.get("llm_metadata") or {}
        usage = llm_meta.get("usage") or {}

        span_input = {"code": entry.get("code")}
        span_output = {
            "stdout": entry.get("stdout"),
            "stderr": entry.get("stderr"),
        }
        span_metadata: dict[str, Any] = {
            "iteration": entry.get("iteration"),
            "terminated": entry.get("terminated", False),
        }

        # LLM telemetry
        if llm_meta.get("model"):
            span_metadata["model"] = llm_meta["model"]
        if usage:
            span_metadata["usage"] = usage
        if llm_meta.get("cost") is not None:
            span_metadata["cost"] = llm_meta["cost"]
        if llm_meta.get("provider"):
            span_metadata["provider"] = llm_meta["provider"]

        span = trace.span(
            name=f"rr_iteration_{entry.get('iteration', '?')}",
            input=span_input,
            output=span_output,
            metadata=span_metadata,
        )
        span.end()

    def _log_subagent_span(
        self, trace: Any, call: dict[str, Any]
    ) -> None:
        """Create a child span for a single sub-agent LLM call."""
        usage = call.get("usage") or {}

        span_input = {
            "question": call.get("question"),
            "context_length": call.get("context_length"),
        }
        span_output = {
            "response_length": call.get("response_length"),
        }
        span_metadata: dict[str, Any] = {
            "call_number": call.get("call_number"),
            "mode": call.get("mode"),
        }
        if call.get("model"):
            span_metadata["model"] = call["model"]
        if usage:
            span_metadata["usage"] = usage
        if call.get("cost") is not None:
            span_metadata["cost"] = call["cost"]

        span = trace.span(
            name=f"subagent_call_{call.get('call_number', '?')}",
            input=span_input,
            output=span_output,
            metadata=span_metadata,
        )
        span.end()

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
        self, reflection: Any, rr_trace: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract output data from the reflection."""
        result: dict[str, Any] = {
            "key_insight": reflection.key_insight,
            "reasoning": reflection.reasoning[:500],
            "learnings_count": len(reflection.extracted_learnings),
        }
        if rr_trace.get("timed_out"):
            result["timed_out"] = True
        return result

    def _build_metadata(
        self,
        rr_trace: dict[str, Any],
        iterations: list[dict[str, Any]],
        subagent_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build metadata dict for the parent trace with aggregated LLM stats."""
        metadata: dict[str, Any] = {
            "total_iterations": rr_trace.get("total_iterations", 0),
        }

        # Aggregate token/cost totals across all LLM calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        model = None

        for entry in iterations:
            llm_meta = entry.get("llm_metadata") or {}
            usage = llm_meta.get("usage") or {}
            total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            total_completion_tokens += usage.get("completion_tokens", 0) or 0
            if llm_meta.get("cost") is not None:
                total_cost += llm_meta["cost"]
            if not model and llm_meta.get("model"):
                model = llm_meta["model"]

        for call in subagent_calls:
            usage = call.get("usage") or {}
            total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            total_completion_tokens += usage.get("completion_tokens", 0) or 0
            if call.get("cost") is not None:
                total_cost += call["cost"]

        if model:
            metadata["model"] = model
        if total_prompt_tokens or total_completion_tokens:
            metadata["total_prompt_tokens"] = total_prompt_tokens
            metadata["total_completion_tokens"] = total_completion_tokens
            metadata["total_tokens"] = total_prompt_tokens + total_completion_tokens
        if total_cost:
            metadata["total_cost"] = total_cost

        if subagent_calls:
            metadata["subagent_call_count"] = len(subagent_calls)

        return metadata

    def flush(self) -> None:
        """Drain buffered traces before process exit."""
        if self._client is not None and hasattr(self._client, "flush"):
            try:
                self._client.flush()
            except Exception:
                pass
