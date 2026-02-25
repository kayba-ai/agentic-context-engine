"""OpikStep — log ACE pipeline traces to Opik.

Terminal side-effect step that creates an Opik trace per sample with
pipeline metadata, agent output, reflection insights, and skill
manager operations.  Optionally registers the LiteLLM OpikLogger
callback for per-LLM-call token/cost tracking.

Place at the end of the pipeline (after ApplyStep).  Gracefully
degrades to a no-op when Opik is not installed or is disabled.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..core.context import ACEStepContext

logger = logging.getLogger(__name__)

# Soft-import Opik — OpikStep is a no-op when the package is absent.
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


def register_opik_litellm_callback(
    project_name: str = "ace-framework",
) -> bool:
    """Register ``OpikLogger`` on ``litellm.callbacks`` for token/cost tracking.

    Standalone helper — call this when you want LiteLLM-level Opik
    tracing without adding ``OpikStep`` to the pipeline.

    Returns ``True`` if the callback was successfully registered.
    """
    if not OPIK_AVAILABLE or _opik_disabled():
        return False
    try:
        import litellm
        from litellm.integrations.opik.opik import OpikLogger

        opik_logger = OpikLogger(project_name=project_name)
        already = any(
            getattr(cb, "__class__", None)
            and cb.__class__.__name__.lower() == "opik"
            for cb in getattr(litellm, "callbacks", [])
        )
        if not already:
            litellm.callbacks.append(opik_logger)
            logger.debug("Opik LiteLLM callback registered for token tracking")
        return True
    except ImportError:
        logger.debug("LiteLLM Opik integration not available")
        return False
    except Exception as exc:
        logger.debug("Failed to register Opik LiteLLM callback: %s", exc)
        return False


class OpikStep:
    """Log ACE pipeline traces to Opik.

    Pure side-effect step — reads context fields and creates an Opik
    trace per sample.  Never mutates the context.

    Args:
        project_name: Opik project name.
        tags: Tags applied to every trace.
        register_litellm_callback: Also register ``OpikLogger`` on
            ``litellm.callbacks`` for per-LLM-call token/cost tracking.
    """

    requires = frozenset({"skillbook"})
    provides = frozenset()

    def __init__(
        self,
        project_name: str = "ace-framework",
        tags: list[str] | None = None,
        register_litellm_callback: bool = True,
    ) -> None:
        self.project_name = project_name
        self.tags = tags or ["ace"]
        self._client: Any | None = None
        self.enabled = OPIK_AVAILABLE and not _opik_disabled()

        if self.enabled:
            try:
                self._client = _opik.Opik(project_name=project_name)
            except Exception as exc:
                logger.debug("OpikStep: failed to create Opik client: %s", exc)
                self.enabled = False

        if self.enabled and register_litellm_callback:
            register_opik_litellm_callback(project_name=project_name)

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if not self.enabled:
            return ctx

        try:
            self._log_trace(ctx)
        except Exception as exc:
            logger.debug("OpikStep: failed to log trace (non-critical): %s", exc)

        return ctx

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_trace(self, ctx: ACEStepContext) -> None:
        """Build and send an Opik trace from the pipeline context."""
        trace_input = self._build_input(ctx)
        trace_output = self._build_output(ctx)
        metadata = self._build_metadata(ctx)
        feedback_scores = self._build_feedback_scores(ctx)
        tags = self.tags + [f"epoch-{ctx.epoch}"]

        trace = self._client.trace(
            name="ace_pipeline",
            input=trace_input,
            output=trace_output,
            metadata=metadata,
            feedback_scores=feedback_scores or None,
            tags=tags,
            project_name=self.project_name,
        )
        trace.end()

    def _build_input(self, ctx: ACEStepContext) -> dict[str, Any]:
        """Extract input data from the context."""
        result: dict[str, Any] = {}

        if isinstance(ctx.trace, dict):
            result["question"] = ctx.trace.get("question", "")
            if ctx.trace.get("context"):
                result["context"] = ctx.trace["context"]
        elif ctx.sample is not None:
            if hasattr(ctx.sample, "question"):
                result["question"] = ctx.sample.question
            elif isinstance(ctx.sample, str):
                result["task"] = ctx.sample
            if hasattr(ctx.sample, "context") and ctx.sample.context:
                result["context"] = ctx.sample.context

        return result

    def _build_output(self, ctx: ACEStepContext) -> dict[str, Any]:
        """Extract output data from the context."""
        result: dict[str, Any] = {}

        if ctx.agent_output:
            result["answer"] = ctx.agent_output.final_answer
            result["reasoning"] = ctx.agent_output.reasoning
            if ctx.agent_output.skill_ids:
                result["skill_ids_cited"] = ctx.agent_output.skill_ids
        elif isinstance(ctx.trace, dict) and ctx.trace.get("answer"):
            result["answer"] = ctx.trace["answer"]

        return result

    def _build_metadata(self, ctx: ACEStepContext) -> dict[str, Any]:
        """Build metadata dict from the context."""
        metadata: dict[str, Any] = {
            "epoch": ctx.epoch,
            "total_epochs": ctx.total_epochs,
            "step_index": ctx.step_index,
            "global_sample_index": ctx.global_sample_index,
            "skill_count": len(ctx.skillbook) if ctx.skillbook else 0,
        }

        if ctx.reflection:
            metadata["key_insight"] = ctx.reflection.key_insight
            metadata["learnings_count"] = len(ctx.reflection.extracted_learnings)

        if ctx.skill_manager_output:
            ops = ctx.skill_manager_output.operations
            metadata["operations_count"] = len(ops)
            op_counts: dict[str, int] = {}
            for op in ops:
                op_counts[op.type] = op_counts.get(op.type, 0) + 1
            if op_counts:
                metadata["operation_types"] = op_counts

        return metadata

    def _build_feedback_scores(self, ctx: ACEStepContext) -> list[dict[str, Any]]:
        """Extract feedback scores from the trace (if environment was used)."""
        scores: list[dict[str, Any]] = []

        if isinstance(ctx.trace, dict):
            feedback = ctx.trace.get("feedback", "")
            if feedback:
                is_correct = feedback.lower().startswith("correct")
                scores.append(
                    {
                        "name": "accuracy",
                        "value": 1.0 if is_correct else 0.0,
                        "reason": feedback,
                    }
                )

        return scores
