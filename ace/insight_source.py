"""Helpers for attaching trace provenance to skillbook update operations."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .core.insight_source import (
    TRACE_IDENTITY_METADATA_KEY,
    InsightSource,
    TraceIdentity,
    TraceReference,
    coerce_trace_identity,
    infer_trace_identity,
)
from .core.outputs import ExtractedLearning, ReflectorOutput
from .core.skillbook import UpdateOperation


def _flatten_learnings(
    reflections: Sequence[ReflectorOutput],
) -> list[tuple[int, ExtractedLearning]]:
    learnings: list[tuple[int, ExtractedLearning]] = []
    for reflection_index, reflection in enumerate(reflections):
        learnings.extend(
            (reflection_index, learning) for learning in reflection.extracted_learnings
        )
    return learnings


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _unique_non_empty(*values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _first_non_empty(value)
        if text is None or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _anchor_tokens(value: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_:-]+", value)
        if len(token) >= 8 or "_" in token or any(char.isdigit() for char in token)
    }
    return tokens


def _match_trace_text(text: str, excerpt: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_excerpt = _normalize_text(excerpt)
    if not normalized_text or not normalized_excerpt:
        return False
    if normalized_excerpt in normalized_text:
        return True
    if len(normalized_text) >= 24 and normalized_text in normalized_excerpt:
        return True

    shared_anchor_tokens = _anchor_tokens(text) & _anchor_tokens(excerpt)
    if any(
        "_" in token or any(char.isdigit() for char in token)
        for token in shared_anchor_tokens
    ):
        return True
    return len(shared_anchor_tokens) >= 2


def _append_path(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key)}]"


def _extract_span_ids(value: Mapping[str, Any]) -> list[str]:
    span_ids: list[str] = []
    for key in ("span_id", "spanId"):
        span_id = value.get(key)
        if span_id is not None:
            span_ids.append(str(span_id))
    for key in ("span_ids", "spanIds"):
        raw_span_ids = value.get(key)
        if isinstance(raw_span_ids, list):
            span_ids.extend(
                str(span_id) for span_id in raw_span_ids if span_id is not None
            )
    return list(dict.fromkeys(span_ids))


def _search_trace_refs(
    value: Any,
    excerpt: str,
    *,
    excerpt_location: str,
    path: str = "$",
    step_indices: tuple[int, ...] = (),
    message_indices: tuple[int, ...] = (),
    inherited_span_ids: tuple[str, ...] = (),
    limit: int = 3,
) -> list[TraceReference]:
    refs: list[TraceReference] = []

    if isinstance(value, Mapping):
        span_ids = tuple(
            dict.fromkeys([*inherited_span_ids, *_extract_span_ids(value)])
        )
        for key, child in value.items():
            child_path = _append_path(path, str(key))

            if key == "steps" and isinstance(child, list):
                for index, item in enumerate(child):
                    refs.extend(
                        _search_trace_refs(
                            item,
                            excerpt,
                            excerpt_location=excerpt_location,
                            path=_append_path(child_path, index),
                            step_indices=(*step_indices, index),
                            message_indices=message_indices,
                            inherited_span_ids=span_ids,
                            limit=limit,
                        )
                    )
                    if len(refs) >= limit:
                        return refs[:limit]
                continue

            if key == "messages" and isinstance(child, list):
                for index, item in enumerate(child):
                    refs.extend(
                        _search_trace_refs(
                            item,
                            excerpt,
                            excerpt_location=excerpt_location,
                            path=_append_path(child_path, index),
                            step_indices=step_indices,
                            message_indices=(*message_indices, index),
                            inherited_span_ids=span_ids,
                            limit=limit,
                        )
                    )
                    if len(refs) >= limit:
                        return refs[:limit]
                continue

            if isinstance(child, (Mapping, list)):
                refs.extend(
                    _search_trace_refs(
                        child,
                        excerpt,
                        excerpt_location=excerpt_location,
                        path=child_path,
                        step_indices=step_indices,
                        message_indices=message_indices,
                        inherited_span_ids=span_ids,
                        limit=limit,
                    )
                )
                if len(refs) >= limit:
                    return refs[:limit]
                continue

            text = _first_non_empty(child)
            if text is None or not _match_trace_text(text, excerpt):
                continue

            refs.append(
                TraceReference(
                    step_indices=list(step_indices),
                    message_indices=list(message_indices),
                    span_ids=list(span_ids),
                    json_path=child_path,
                    text_excerpt=text[:500],
                    excerpt_location=excerpt_location,
                )
            )
            if len(refs) >= limit:
                return refs[:limit]
        return refs[:limit]

    if isinstance(value, list):
        for index, item in enumerate(value):
            refs.extend(
                _search_trace_refs(
                    item,
                    excerpt,
                    excerpt_location=excerpt_location,
                    path=_append_path(path, index),
                    step_indices=step_indices,
                    message_indices=message_indices,
                    inherited_span_ids=inherited_span_ids,
                    limit=limit,
                )
            )
            if len(refs) >= limit:
                return refs[:limit]
        return refs[:limit]

    text = _first_non_empty(value)
    if text is None or not _match_trace_text(text, excerpt):
        return []

    return [
        TraceReference(
            step_indices=list(step_indices),
            message_indices=list(message_indices),
            span_ids=list(inherited_span_ids),
            json_path=path,
            text_excerpt=text[:500],
            excerpt_location=excerpt_location,
        )
    ]


def _looks_like_combined_steps_batch(trace: Mapping[str, Any]) -> bool:
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    return all(
        isinstance(step, Mapping)
        and step.get("role") == "conversation"
        and isinstance(step.get("content"), Mapping)
        for step in steps
    )


def _get_batch_items(trace: Any) -> list[Any] | None:
    if isinstance(trace, list):
        return trace
    if not isinstance(trace, Mapping):
        return None
    for key in ("items", "tasks"):
        items = trace.get(key)
        if isinstance(items, list):
            return items
    if _looks_like_combined_steps_batch(trace):
        steps = trace.get("steps")
        if isinstance(steps, list):
            return steps
    return None


def _extract_batch_item_payload(item: Any) -> Any:
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


def _get_batch_item_id(item: Any, index: int) -> str:
    payload = _extract_batch_item_payload(item)
    if isinstance(item, Mapping):
        for key in ("trace_id", "sample_id", "item_id", "task_id", "id"):
            value = item.get(key)
            if value is not None:
                return str(value)
    if isinstance(payload, Mapping):
        for key in ("trace_id", "sample_id", "item_id", "task_id", "id"):
            value = payload.get(key)
            if value is not None:
                return str(value)
    return f"item_{index}"


def _collect_text_fragments(value: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(value, Mapping):
        for child in value.values():
            fragments.extend(_collect_text_fragments(child))
    elif isinstance(value, list):
        for child in value:
            fragments.extend(_collect_text_fragments(child))
    else:
        text = _first_non_empty(value)
        if text is not None:
            fragments.append(text)
    return fragments


def _operation_text(operation: UpdateOperation) -> str:
    return "\n".join(
        text
        for text in (
            _first_non_empty(operation.content),
            _first_non_empty(operation.justification),
            _first_non_empty(operation.evidence),
        )
        if text is not None
    )


def _operation_suggests_multiple_sources(operation: UpdateOperation) -> bool:
    operation_text = _normalize_text(_operation_text(operation))
    if not operation_text:
        return False
    return bool(
        re.search(
            r"\b(both|all|multiple|combined|across|together|shared|common)\b",
            operation_text,
        )
    )


def _valid_reflection_indices(
    operation: UpdateOperation,
    reflections: Sequence[ReflectorOutput],
) -> list[int]:
    indices: list[int] = []
    for index in operation.reflection_indices:
        if 0 <= index < len(reflections) and index not in indices:
            indices.append(index)
    return indices


def _batch_item_match_score(operation: UpdateOperation, item: Any, index: int) -> int:
    operation_text = _operation_text(operation)
    if not operation_text:
        return 0

    item_id = _get_batch_item_id(item, index)
    item_fragments = _collect_text_fragments(item)
    item_text = "\n".join(item_fragments)
    if not item_text:
        return 0

    score = 0
    normalized_operation = _normalize_text(operation_text)
    normalized_item_id = _normalize_text(item_id)
    if normalized_item_id and normalized_item_id in normalized_operation:
        score += 20

    item_suffix = re.search(r"(\d+)$", item_id)
    if item_suffix is not None:
        item_number = item_suffix.group(1)
        if re.search(
            rf"\b(?:task|item|reflection)\s*{re.escape(item_number)}\b",
            normalized_operation,
        ):
            score += 20

    shared_anchor_tokens = _anchor_tokens(operation_text) & _anchor_tokens(item_text)
    score += 5 * len(shared_anchor_tokens)

    operation_tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9_:-]+", normalized_operation)
        if len(token) >= 5
    }
    item_tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9_:-]+", _normalize_text(item_text))
        if len(token) >= 5
    }
    score += len(operation_tokens & item_tokens)
    return score


def _best_matching_batch_index(
    operation: UpdateOperation,
    batch_items: Sequence[Any],
) -> int | None:
    scores = [
        _batch_item_match_score(operation, item, index)
        for index, item in enumerate(batch_items)
    ]
    if not scores:
        return None
    best_score = max(scores)
    if best_score <= 0:
        return None
    if scores.count(best_score) > 1:
        return None
    return scores.index(best_score)


def _matching_batch_indices(
    operation: UpdateOperation,
    batch_items: Sequence[Any],
) -> list[int]:
    scores = [
        _batch_item_match_score(operation, item, index)
        for index, item in enumerate(batch_items)
    ]
    if not scores:
        return []

    best_score = max(scores)
    if best_score <= 0:
        return []

    strong_matches = [
        index for index, score in enumerate(scores) if score >= max(20, best_score - 5)
    ]
    if len(strong_matches) >= 2:
        return strong_matches

    if _operation_suggests_multiple_sources(operation):
        relaxed_matches = [
            index
            for index, score in enumerate(scores)
            if score >= max(1, best_score // 2)
        ]
        if len(relaxed_matches) >= 2:
            return relaxed_matches

    return []


def _prune_explicit_batch_indices(
    operation: UpdateOperation,
    batch_items: Sequence[Any],
    indices: Sequence[int],
) -> list[int]:
    if len(indices) <= 1:
        return list(indices)

    scores = {
        index: _batch_item_match_score(operation, batch_items[index], index)
        for index in indices
    }
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return list(indices)

    threshold = max(1, best_score // 2)
    pruned = [index for index in indices if scores.get(index, 0) >= threshold]
    return pruned or list(indices)


def _get_reflection_item_id(reflection: ReflectorOutput | None) -> str | None:
    if reflection is None:
        return None
    for key in ("trace_id", "sample_id", "item_id", "task_id", "id"):
        value = reflection.raw.get(key)
        if value is not None:
            return str(value)
    return None


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


def _resolve_operation_reflection(
    operation: UpdateOperation,
    reflections: Sequence[ReflectorOutput],
) -> tuple[int | None, ReflectorOutput | None, ExtractedLearning | None]:
    if not reflections:
        return None, None, None

    explicit_reflection_indices = _valid_reflection_indices(operation, reflections)
    reflection_index = operation.reflection_index
    if reflection_index is not None and not 0 <= reflection_index < len(reflections):
        reflection_index = None
    if reflection_index is None and explicit_reflection_indices:
        reflection_index = explicit_reflection_indices[0]

    flattened_learnings = _flatten_learnings(reflections)

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
            elif 0 <= operation.learning_index < len(flattened_learnings):
                reflection_index = flattened_learnings[operation.learning_index][0]

    reflection = None
    if reflection_index is not None:
        reflection = reflections[reflection_index]

    learning = None
    if (
        reflection is not None
        and operation.learning_index is not None
        and 0 <= operation.learning_index < len(reflection.extracted_learnings)
    ):
        learning = reflection.extracted_learnings[operation.learning_index]
    elif operation.learning_index is not None and 0 <= operation.learning_index < len(
        flattened_learnings
    ):
        flattened_reflection_index, learning = flattened_learnings[
            operation.learning_index
        ]
        if reflection is None:
            reflection_index = flattened_reflection_index
            reflection = reflections[flattened_reflection_index]

    return reflection_index, reflection, learning


def _primary_batch_index(
    operation: UpdateOperation,
    batch_items: Sequence[Any],
    reflection_index: int | None,
    reflection: ReflectorOutput | None,
) -> int | None:
    heuristic_index = _best_matching_batch_index(operation, batch_items)

    reflection_item_id = _get_reflection_item_id(reflection)
    if reflection_item_id is not None:
        for index, item in enumerate(batch_items):
            if _get_batch_item_id(item, index) == reflection_item_id:
                if heuristic_index is not None and heuristic_index != index:
                    heuristic_score = _batch_item_match_score(
                        operation,
                        batch_items[heuristic_index],
                        heuristic_index,
                    )
                    reflection_score = _batch_item_match_score(operation, item, index)
                    if heuristic_score > reflection_score:
                        return heuristic_index
                return index

    if reflection_index is not None and 0 <= reflection_index < len(batch_items):
        if heuristic_index is not None and heuristic_index != reflection_index:
            heuristic_score = _batch_item_match_score(
                operation,
                batch_items[heuristic_index],
                heuristic_index,
            )
            reflection_score = _batch_item_match_score(
                operation,
                batch_items[reflection_index],
                reflection_index,
            )
            if heuristic_score > reflection_score:
                return heuristic_index
        return reflection_index

    return heuristic_index


def _operation_batch_indices(
    trace: Any,
    operation: UpdateOperation,
    reflections: Sequence[ReflectorOutput],
    reflection_index: int | None,
    reflection: ReflectorOutput | None,
) -> list[int]:
    batch_items = _get_batch_items(trace)
    if batch_items is None:
        return []

    selected_indices: list[int] = []
    explicit_indices = _prune_explicit_batch_indices(
        operation,
        batch_items,
        _valid_reflection_indices(operation, reflections),
    )
    for index in explicit_indices:
        if 0 <= index < len(batch_items) and index not in selected_indices:
            selected_indices.append(index)

    primary_index = _primary_batch_index(
        operation,
        batch_items,
        reflection_index,
        reflection,
    )
    if primary_index is not None and primary_index not in selected_indices:
        selected_indices.append(primary_index)

    for index in _matching_batch_indices(operation, batch_items):
        if index not in selected_indices:
            selected_indices.append(index)

    return selected_indices


def _trace_question(trace: Any) -> str | None:
    if isinstance(trace, Mapping):
        nested_trace = trace.get("trace")
        if isinstance(nested_trace, Mapping):
            return _first_non_empty(trace.get("question"), nested_trace.get("question"))
        return _first_non_empty(trace.get("question"))
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


def _build_trace_refs(
    *,
    trace: Any,
    operation: UpdateOperation,
    learning: ExtractedLearning | None,
) -> list[TraceReference]:
    excerpt_location = (
        "operation.evidence" if operation.evidence else "reflection.learning"
    )
    search_terms = _unique_non_empty(
        operation.evidence,
        getattr(learning, "evidence", None),
        getattr(learning, "learning", None),
    )

    for term in search_terms:
        refs = _search_trace_refs(
            trace,
            term,
            excerpt_location=excerpt_location,
        )
        if refs:
            return refs

    excerpt = search_terms[0] if search_terms else None
    if excerpt is None:
        return []

    return [
        TraceReference(
            text_excerpt=excerpt[:500],
            excerpt_location=excerpt_location,
        )
    ]


def _resolve_learning_for_reflection(
    operation: UpdateOperation,
    reflection: ReflectorOutput | None,
) -> ExtractedLearning | None:
    if (
        reflection is not None
        and operation.learning_index is not None
        and 0 <= operation.learning_index < len(reflection.extracted_learnings)
    ):
        return reflection.extracted_learnings[operation.learning_index]
    return None


def _source_signature(source: InsightSource) -> str:
    return json.dumps(source.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


def build_insight_source(
    *,
    sample_question: str = "",
    epoch: int | None = None,
    step: int | None = None,
    error_identification: str = "",
    agent_output: Any | None = None,
    reflection: ReflectorOutput | None = None,
    reflections: Sequence[ReflectorOutput] | None = None,
    operations: list[UpdateOperation],
    trace: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    trace_identity: TraceIdentity | Mapping[str, Any] | None = None,
    sample: Any | None = None,
    sample_id: str | None = None,
    relation: str = "seed",
) -> list[UpdateOperation]:
    """Attach provenance to update operations in place.

    Returns the mutated *operations* list for convenience.
    """
    del agent_output  # accepted for hosted/backward-compatible call sites

    if not operations:
        return operations

    reflection_seq: Sequence[ReflectorOutput]
    if reflections is not None:
        reflection_seq = reflections
    elif reflection is not None:
        reflection_seq = (reflection,)
    else:
        reflection_seq = ()

    parent_identity = _resolve_trace_identity(
        trace=trace,
        sample=sample,
        metadata=metadata,
        trace_identity=trace_identity,
    )

    fallback_error = _first_non_empty(
        error_identification,
        *[item.error_identification for item in reflection_seq],
    )
    fallback_question = _first_non_empty(
        sample_question,
        _trace_question(trace),
        getattr(sample, "question", None),
    )

    for operation in operations:
        if operation.insight_source is not None:
            continue

        reflection_index, matched_reflection, learning = _resolve_operation_reflection(
            operation,
            reflection_seq,
        )
        batch_items = _get_batch_items(trace)
        source_entries: list[
            tuple[int | None, Any, ReflectorOutput | None, ExtractedLearning | None]
        ] = []
        if batch_items is None:
            source_entries.append((None, trace, matched_reflection, learning))
        else:
            batch_indices = _operation_batch_indices(
                trace,
                operation,
                reflection_seq,
                reflection_index,
                matched_reflection,
            )
            if not batch_indices:
                source_entries.append((None, trace, matched_reflection, learning))
            else:
                for batch_index in batch_indices:
                    source_reflection_index, source_reflection = (
                        _reflection_for_batch_index(
                            reflection_seq,
                            batch_items,
                            batch_index,
                        )
                    )
                    source_learning = (
                        learning
                        if source_reflection_index == reflection_index
                        else _resolve_learning_for_reflection(
                            operation, source_reflection
                        )
                    )
                    source_entries.append(
                        (
                            batch_index,
                            batch_items[batch_index],
                            source_reflection,
                            source_learning,
                        )
                    )

        sources: list[InsightSource] = []
        seen_signatures: set[str] = set()
        primary_batch_index = source_entries[0][0] if source_entries else None
        for (
            batch_index,
            operation_trace,
            source_reflection,
            source_learning,
        ) in source_entries:
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

            effective_error = _first_non_empty(
                error_identification,
                getattr(source_reflection, "error_identification", None),
                getattr(matched_reflection, "error_identification", None),
                fallback_error,
            )
            effective_question = _first_non_empty(
                _trace_question(operation_trace),
                sample_question,
                getattr(sample, "question", None),
                fallback_question,
            )
            effective_sample_id = sample_id or identity.trace_id
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
                step=step,
                learning_index=operation.learning_index,
                learning_text=getattr(source_learning, "learning", None),
                error_identification=effective_error,
                operation_type=operation.type,
                trace_refs=_build_trace_refs(
                    trace=operation_trace,
                    operation=operation,
                    learning=source_learning,
                ),
                sample_id=effective_sample_id,
            )
            signature = _source_signature(source)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            sources.append(source)

        if not sources:
            continue
        operation.insight_source = sources[0] if len(sources) == 1 else sources

    return operations


__all__ = [
    "TRACE_IDENTITY_METADATA_KEY",
    "InsightSource",
    "TraceIdentity",
    "TraceReference",
    "build_insight_source",
    "infer_trace_identity",
]
