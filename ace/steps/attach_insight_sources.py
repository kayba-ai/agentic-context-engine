"""AttachInsightSourcesStep — enrich update operations with trace provenance."""

from __future__ import annotations

from copy import deepcopy

from ..core.context import ACEStepContext
from ..core.skillbook import UpdateBatch
from ..insight_source import build_insight_source


class AttachInsightSourcesStep:
    """Attach provenance metadata to update operations.

    Pure step — reads the current trace, reflection output, and update batch,
    then returns a new ``UpdateBatch`` with ``insight_source`` attached to
    operations that do not already have one.
    """

    requires = frozenset({"trace", "reflections", "skill_manager_output", "metadata"})
    provides = frozenset({"skill_manager_output"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        batch = ctx.skill_manager_output
        if batch is None or not batch.operations:
            return ctx

        operations = deepcopy(batch.operations)
        build_insight_source(
            sample_question=self._sample_question(ctx),
            epoch=ctx.epoch,
            step=ctx.step_index,
            reflections=ctx.reflections,
            operations=operations,
            trace=ctx.trace,
            metadata=ctx.metadata,
            sample=ctx.sample,
            sample_id=self._sample_id(ctx),
        )

        return ctx.replace(
            skill_manager_output=UpdateBatch(
                reasoning=batch.reasoning,
                operations=operations,
            )
        )

    @staticmethod
    def _sample_question(ctx: ACEStepContext) -> str:
        if isinstance(ctx.trace, dict):
            question = ctx.trace.get("question")
            if isinstance(question, str) and question.strip():
                return question
        question = getattr(ctx.sample, "question", None)
        if isinstance(question, str):
            return question
        return ""

    @staticmethod
    def _sample_id(ctx: ACEStepContext) -> str | None:
        if isinstance(ctx.trace, dict):
            sample_id = ctx.trace.get("sample_id")
            if sample_id is not None:
                text = str(sample_id).strip()
                if text:
                    return text
        sample_id = getattr(ctx.sample, "id", None)
        if sample_id is not None:
            text = str(sample_id).strip()
            if text:
                return text
        return None
