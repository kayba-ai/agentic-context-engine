"""Typed provenance models for skillbook insight sources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

TRACE_IDENTITY_METADATA_KEY = "ace.trace_identity"


def make_trace_uid(source_system: str, trace_id: str) -> str:
    """Return a stable composite identifier for a trace."""
    return f"{source_system}:{trace_id}"


def fingerprint_trace(value: Any) -> str:
    """Return a stable content fingerprint for a trace-like object."""
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _legacy_trace_id(payload: Mapping[str, Any]) -> str | None:
    for key in ("sample_id", "item_id", "task_id", "id"):
        value = _coerce_str(payload.get(key))
        if value is not None:
            return value
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


@dataclass
class TraceIdentity:
    """Stable identity for a trace across storage and UI layers."""

    source_system: str
    trace_id: str
    display_name: str | None = None
    trace_uid: str | None = None

    def __post_init__(self) -> None:
        self.source_system = self.source_system.strip() or "local"
        self.trace_id = self.trace_id.strip()
        if not self.trace_uid:
            self.trace_uid = make_trace_uid(self.source_system, self.trace_id)
        if self.display_name is not None:
            self.display_name = self.display_name.strip() or None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceIdentity":
        trace_uid = _coerce_str(payload.get("trace_uid"))
        source_system = _coerce_str(payload.get("source_system"))
        trace_id = _coerce_str(payload.get("trace_id"))
        if (
            trace_uid
            and (source_system is None or trace_id is None)
            and ":" in trace_uid
        ):
            inferred_source, inferred_id = trace_uid.split(":", 1)
            source_system = source_system or inferred_source
            trace_id = trace_id or inferred_id
        if trace_id is None:
            legacy_id = _legacy_trace_id(payload)
            if legacy_id is not None:
                trace_id = legacy_id
        if source_system is None:
            source_system = "legacy"
        if trace_id is None:
            trace_id = fingerprint_trace(dict(payload))
        return cls(
            source_system=source_system,
            trace_id=trace_id,
            display_name=_coerce_str(payload.get("display_name"))
            or _legacy_trace_id(payload)
            or trace_id,
            trace_uid=trace_uid,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        data["trace_uid"] = self.trace_uid
        data["source_system"] = self.source_system
        data["trace_id"] = self.trace_id
        if self.display_name is not None:
            data["display_name"] = self.display_name
        return data


@dataclass
class InsightSource:
    """A single provenance record describing how a trace informed a skill."""

    trace_uid: str
    source_system: str
    trace_id: str
    display_name: str | None = None
    relation: str | None = None
    sample_question: str | None = None
    epoch: int | None = None
    operation_type: str | None = None
    error_identification: str | None = None
    learning_text: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InsightSource":
        identity = TraceIdentity.from_dict(payload)
        return cls(
            trace_uid=identity.trace_uid
            or make_trace_uid(identity.source_system, identity.trace_id),
            source_system=identity.source_system,
            trace_id=identity.trace_id,
            display_name=identity.display_name,
            relation=_coerce_str(payload.get("relation")),
            sample_question=_coerce_str(payload.get("sample_question")),
            epoch=_safe_int(payload.get("epoch")),
            operation_type=_coerce_str(payload.get("operation_type")),
            error_identification=_coerce_str(payload.get("error_identification")),
            learning_text=_coerce_str(payload.get("learning_text")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "trace_uid": self.trace_uid,
            "source_system": self.source_system,
            "trace_id": self.trace_id,
        }
        if self.display_name is not None:
            data["display_name"] = self.display_name
        if self.relation is not None:
            data["relation"] = self.relation
        if self.sample_question is not None:
            data["sample_question"] = self.sample_question
        if self.epoch is not None:
            data["epoch"] = self.epoch
        if self.operation_type is not None:
            data["operation_type"] = self.operation_type
        if self.error_identification is not None:
            data["error_identification"] = self.error_identification
        if self.learning_text is not None:
            data["learning_text"] = self.learning_text
        return data


def coerce_trace_identity(value: TraceIdentity | Mapping[str, Any]) -> TraceIdentity:
    if isinstance(value, TraceIdentity):
        return value
    return TraceIdentity.from_dict(value)


def coerce_insight_source(value: InsightSource | Mapping[str, Any]) -> InsightSource:
    if isinstance(value, InsightSource):
        return value
    return InsightSource.from_dict(value)


def coerce_insight_sources(value: Any) -> list[InsightSource]:
    if value is None:
        return []
    if isinstance(value, InsightSource):
        return [value]
    if isinstance(value, Mapping):
        return [InsightSource.from_dict(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        sources: list[InsightSource] = []
        for item in value:
            if isinstance(item, InsightSource):
                sources.append(item)
            elif isinstance(item, Mapping):
                sources.append(InsightSource.from_dict(item))
        return sources
    return []


def infer_trace_identity(
    *,
    trace: Any = None,
    sample: Any = None,
    metadata: Mapping[str, Any] | None = None,
    default_source_system: str = "local",
) -> TraceIdentity:
    """Infer the best available stable trace identity."""
    if metadata:
        raw_metadata_identity = metadata.get(
            TRACE_IDENTITY_METADATA_KEY
        ) or metadata.get("trace_identity")
        if isinstance(raw_metadata_identity, (TraceIdentity, Mapping)):
            return coerce_trace_identity(raw_metadata_identity)

    sample_metadata = getattr(sample, "metadata", None)
    if isinstance(sample_metadata, Mapping):
        raw_sample_identity = sample_metadata.get(
            TRACE_IDENTITY_METADATA_KEY
        ) or sample_metadata.get("trace_identity")
        if isinstance(raw_sample_identity, (TraceIdentity, Mapping)):
            return coerce_trace_identity(raw_sample_identity)
        trace_id = _coerce_str(sample_metadata.get("trace_id")) or _legacy_trace_id(
            sample_metadata
        )
        source_system = _coerce_str(sample_metadata.get("source_system"))
        display_name = _coerce_str(sample_metadata.get("display_name"))
        trace_uid = _coerce_str(sample_metadata.get("trace_uid"))
        if trace_uid or trace_id:
            if trace_id is None and trace_uid and ":" in trace_uid:
                inferred_source, inferred_id = trace_uid.split(":", 1)
                source_system = source_system or inferred_source
                trace_id = inferred_id
            if trace_id is not None:
                return TraceIdentity(
                    source_system=source_system or "sample",
                    trace_id=trace_id,
                    display_name=display_name
                    or _legacy_trace_id(sample_metadata)
                    or _coerce_str(getattr(sample, "id", None))
                    or trace_id,
                    trace_uid=trace_uid,
                )

    if isinstance(trace, Mapping):
        raw_identity = trace.get(TRACE_IDENTITY_METADATA_KEY) or trace.get(
            "trace_identity"
        )
        if isinstance(raw_identity, (TraceIdentity, Mapping)):
            return coerce_trace_identity(raw_identity)

        if any(key in trace for key in ("trace_uid", "trace_id", "source_system")):
            return TraceIdentity.from_dict(trace)

        legacy_id = _legacy_trace_id(trace)
        if legacy_id is not None:
            return TraceIdentity(
                source_system=_coerce_str(trace.get("source_system"))
                or default_source_system,
                trace_id=legacy_id,
                display_name=_coerce_str(trace.get("display_name"))
                or _coerce_str(trace.get("question"))
                or legacy_id,
            )

    sample_id = _coerce_str(getattr(sample, "id", None))
    if sample_id is not None:
        return TraceIdentity(
            source_system="sample",
            trace_id=sample_id,
            display_name=sample_id,
        )

    fallback_source = (
        trace if trace is not None else getattr(sample, "question", sample)
    )
    fallback_id = fingerprint_trace(fallback_source)

    display_name = None
    if isinstance(trace, Mapping):
        display_name = _coerce_str(trace.get("question")) or _coerce_str(
            trace.get("sample_id")
        )
        if display_name is None:
            display_name = _legacy_trace_id(trace)
    if display_name is None:
        display_name = _coerce_str(getattr(sample, "question", None)) or sample_id

    return TraceIdentity(
        source_system=default_source_system,
        trace_id=fallback_id,
        display_name=display_name or fallback_id,
    )
