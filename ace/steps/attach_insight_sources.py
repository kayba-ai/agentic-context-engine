"""AttachInsightSourcesStep — enrich update operations with trace provenance."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from ..core.context import ACEStepContext
from ..core.insight_source import (
    InsightSource,
    TraceIdentity,
    coerce_trace_identity,
    infer_trace_identity,
)
from ..core.outputs import ExtractedLearning, ReflectorOutput
from ..core.skillbook import UpdateBatch, UpdateOperation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _resolve_trace_identity(
    *,
    trace: Any,
    sample: Any | None,
    metadata: Mapping[str, Any] | None,
    trace_identity: TraceIdentity | Mapping[str, Any] | None,
    default_source_system: str = "local",
) -> TraceIdentity:
    if trace_identity is not None:
        return coerce_trace_identity(trace_identity)
    return infer_trace_identity(
        trace=trace,
        sample=sample,
        metadata=metadata,
        default_source_system=default_source_system,
    )


def _trace_question(trace: Any) -> str | None:
    if isinstance(trace, Mapping):
        nested_trace = trace.get("trace")
        if isinstance(nested_trace, Mapping):
            return _first_non_empty(trace.get("question"), nested_trace.get("question"))
        return _first_non_empty(trace.get("question"))
    return None


# ---------------------------------------------------------------------------
# Batch helpers — deterministic matching only
# ---------------------------------------------------------------------------


def _get_batch_items(trace: Any) -> list[Any] | None:
    """Return the list of batch items from a trace, or None if not a batch."""
    if isinstance(trace, list):
        return trace
    if not isinstance(trace, Mapping):
        return None
    for key in ("items", "tasks"):
        items = trace.get(key)
        if isinstance(items, list):
            return items
    # Combined-steps batch format
    steps = trace.get("steps")
    if isinstance(steps, list) and steps and all(
        isinstance(step, Mapping)
        and step.get("role") == "conversation"
        and isinstance(step.get("content"), Mapping)
        for step in steps
    ):
        return steps
    return None


def _get_batch_item_id(item: Any, index: int) -> str:
    """Extract a stable ID from a batch item."""
    if isinstance(item, Mapping):
        for key in ("trace_id", "sample_id", "item_id", "task_id", "id"):
            value = item.get(key)
            if value is not None:
                return str(value)
        # Check nested content for combined-steps format
        content = item.get("content")
        if isinstance(content, Mapping):
            for key in ("trace_id", "sample_id", "item_id", "task_id", "id"):
                value = content.get(key)
                if value is not None:
                    return str(value)
    return f"item_{index}"


def _extract_batch_item_payload(item: Any) -> Any:
    """Unwrap a batch item to its trace-like payload."""
    if (
        isinstance(item, Mapping)
        and item.get("role") == "conversation"
        and isinstance(item.get("content"), Mapping)
    ):
        return item["content"]
    if isinstance(item, Mapping):
        trace_value = item.get("trace")
        if isinstance(trace_value, (Mapping, list)):
            return trace_value
    return item


def _get_reflection_item_id(reflection: ReflectorOutput | None) -> str | None:
    if reflection is None:
        return None
    for key in ("trace_id", "sample_id", "item_id", "task_id", "id"):
        value = reflection.raw.get(key)
        if value is not None:
            return str(value)
    return None


def _valid_reflection_indices(
    operation: UpdateOperation,
    reflections: Sequence[ReflectorOutput],
) -> list[int]:
    indices: list[int] = []
    for index in operation.reflection_indices:
        if 0 <= index < len(reflections) and index not in indices:
            indices.append(index)
    return indices


def _match_batch_indices_for_operation(
    operation: UpdateOperation,
    batch_items: Sequence[Any],
    reflections: Sequence[ReflectorOutput],
    reflection_index: int | None,
    matched_reflection: ReflectorOutput | None,
) -> list[int]:
    """Deterministic batch matching: explicit indices, ID matching, positional."""
    selected: list[int] = []

    # 1. Explicit reflection_indices → batch indices
    for ri in _valid_reflection_indices(operation, reflections):
        if 0 <= ri < len(batch_items) and ri not in selected:
            selected.append(ri)

    # 2. Match via reflection item_id → batch item_id
    reflection_item_id = _get_reflection_item_id(matched_reflection)
    if reflection_item_id is not None:
        for idx, item in enumerate(batch_items):
            if _get_batch_item_id(item, idx) == reflection_item_id and idx not in selected:
                selected.append(idx)

    # 3. Positional fallback from reflection_index
    if not selected and reflection_index is not None and 0 <= reflection_index < len(batch_items):
        selected.append(reflection_index)

    return selected


# ---------------------------------------------------------------------------
# Reflection resolution
# ---------------------------------------------------------------------------


def _resolve_operation_reflection(
    operation: UpdateOperation,
    reflections: Sequence[ReflectorOutput],
) -> tuple[int | None, ReflectorOutput | None, ExtractedLearning | None]:
    """Determine which reflection and learning an operation is associated with."""
    if not reflections:
        return None, None, None

    explicit_indices = _valid_reflection_indices(operation, reflections)
    reflection_index = operation.reflection_index
    if reflection_index is not None and not 0 <= reflection_index < len(reflections):
        reflection_index = None
    if reflection_index is None and explicit_indices:
        reflection_index = explicit_indices[0]

    if reflection_index is None:
        if len(reflections) == 1:
            reflection_index = 0
        elif operation.learning_index is not None:
            local_candidates = [
                index
                for index, reflection in enumerate(reflections)
                if 0 <= operation.learning_index < len(reflection.extracted_learnings)
            ]
            if len(local_candidates) == 1:
                reflection_index = local_candidates[0]

    reflection = reflections[reflection_index] if reflection_index is not None else None

    learning = None
    if (
        reflection is not None
        and operation.learning_index is not None
        and 0 <= operation.learning_index < len(reflection.extracted_learnings)
    ):
        learning = reflection.extracted_learnings[operation.learning_index]

    return reflection_index, reflection, learning


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------


def _source_signature(source: InsightSource) -> str:
    return json.dumps(source.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


def _reflection_for_batch_index(
    reflections: Sequence[ReflectorOutput],
    batch_items: Sequence[Any],
    batch_index: int,
) -> tuple[int | None, ReflectorOutput | None]:
    item = batch_items[batch_index]
    item_id = _get_batch_item_id(item, batch_index)
    for reflection_index, reflection in enumerate(reflections):
        if _get_reflection_item_id(reflection) == item_id:
            return reflection_index, reflection
    if 0 <= batch_index < len(reflections):
        return batch_index, reflections[batch_index]
    return None, None


def build_insight_source(
    *,
    sample_question: str = "",
    epoch: int | None = None,
    reflections: Sequence[ReflectorOutput] = (),
    operations: list[UpdateOperation],
    trace: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    trace_identity: TraceIdentity | Mapping[str, Any] | None = None,
    sample: Any | None = None,
    sample_id: str | None = None,
    relation: str = "seed",
) -> list[UpdateOperation]:
    """Build and attach provenance to update operations.

    Returns a *new* list of operations with ``insight_source`` populated on
    operations that did not already have one.  The input list is not mutated.

    Provenance records which trace, which epoch, what task, plus the
    Reflector's error_identification and learning_text for downstream
    trace highlighting.
    """
    if not operations:
        return list(operations)

    operations = [deepcopy(op) for op in operations]

    parent_identity = _resolve_trace_identity(
        trace=trace,
        sample=sample,
        metadata=metadata,
        trace_identity=trace_identity,
    )

    fallback_question = _first_non_empty(
        sample_question,
        _trace_question(trace),
        getattr(sample, "question", None),
    )
    fallback_error = _first_non_empty(
        *[item.error_identification for item in reflections],
    )

    for operation in operations:
        if operation.insight_source is not None:
            continue

        reflection_index, matched_reflection, learning = _resolve_operation_reflection(
            operation,
            reflections,
        )
        batch_items = _get_batch_items(trace)
        # Each entry: (batch_index, trace_for_identity)
        source_entries: list[tuple[int | None, Any]] = []

        if batch_items is None:
            source_entries.append((None, trace))
        else:
            batch_indices = _match_batch_indices_for_operation(
                operation,
                batch_items,
                reflections,
                reflection_index,
                matched_reflection,
            )
            if not batch_indices:
                source_entries.append((None, trace))
            else:
                for bidx in batch_indices:
                    source_entries.append((bidx, batch_items[bidx]))

        sources: list[InsightSource] = []
        seen_signatures: set[str] = set()
        primary_batch_index = source_entries[0][0] if source_entries else None

        effective_error = _first_non_empty(
            getattr(matched_reflection, "error_identification", None),
            fallback_error,
        )
        effective_learning = getattr(learning, "learning", None)

        for batch_index, operation_trace in source_entries:
            identity = (
                parent_identity
                if operation_trace is trace
                else _resolve_trace_identity(
                    trace=operation_trace,
                    sample=None,
                    metadata=None,
                    trace_identity=None,
                    default_source_system=parent_identity.source_system,
                )
            )

            effective_question = _first_non_empty(
                _trace_question(operation_trace),
                sample_question,
                getattr(sample, "question", None),
                fallback_question,
            )
            source_relation = (
                relation
                if batch_index is None or batch_index == primary_batch_index
                else "supporting"
            )

            source = InsightSource(
                trace_uid=identity.trace_uid
                or f"{identity.source_system}:{identity.trace_id}",
                source_system=identity.source_system,
                trace_id=identity.trace_id,
                display_name=identity.display_name,
                relation=source_relation,
                sample_question=effective_question,
                epoch=epoch,
                operation_type=operation.type,
                error_identification=effective_error,
                learning_text=effective_learning,
            )
            signature = _source_signature(source)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            sources.append(source)

        if not sources:
            continue
        if len(sources) == 1:
            operation.insight_source = sources[0]
        else:
            operation.insight_source = sources  # type: ignore[assignment]

    return operations


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


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

        enriched_operations = build_insight_source(
            sample_question=self._sample_question(ctx),
            epoch=ctx.epoch,
            reflections=ctx.reflections,
            operations=batch.operations,
            trace=ctx.trace,
            metadata=ctx.metadata,
            sample=ctx.sample,
            sample_id=self._sample_id(ctx),
        )

        return ctx.replace(
            skill_manager_output=UpdateBatch(
                reasoning=batch.reasoning,
                operations=enriched_operations,
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
