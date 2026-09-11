"""Skill, Skillbook, and update operations for the ACE framework."""

from __future__ import annotations

import copy
import json
import re
import threading
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Optional, Union, cast

from .insight_source import InsightSource, coerce_insight_source, coerce_insight_sources

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

OperationType = Literal["ADD", "UPDATE", "TAG", "REMOVE"]
SkillSection = Literal["context", "harness"]
SCHEMA_VERSION = "2"
VALID_SECTIONS: frozenset[str] = frozenset({"context", "harness"})
DEFAULT_LEGACY_SECTION: SkillSection = "context"
InsightSourceInput = Union[
    InsightSource,
    Dict[str, Any],
    List[Union[InsightSource, Dict[str, Any]]],
]
_UNSET = object()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_required_text(value: Any, field_name: str) -> str:
    text = _normalize_optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required and must be non-empty")
    return text


def _normalize_keyword(value: Any) -> str | None:
    text = _normalize_optional_text(value)
    if text is None:
        return None
    text = re.sub(r"\s+", "_", text.lower())
    return text or None


def _normalize_keywords(keywords: Iterable[Any] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    if keywords is None:
        return normalized
    for value in keywords:
        keyword = _normalize_keyword(value)
        if keyword is None or keyword in seen:
            continue
        normalized.append(keyword)
        seen.add(keyword)
    return normalized


def _coerce_section_and_keywords(
    section: str,
    keywords: Iterable[Any] | None,
) -> tuple[SkillSection, list[str]]:
    normalized_section = _normalize_required_text(section, "section").lower()
    normalized_keywords = _normalize_keywords(keywords)
    if normalized_section in VALID_SECTIONS:
        return cast(SkillSection, normalized_section), normalized_keywords

    # Backward-compatible coercion for legacy free-form sections.
    legacy_keywords = _normalize_keywords([normalized_section, *normalized_keywords])
    return DEFAULT_LEGACY_SECTION, legacy_keywords


def _embedding_sidecar_path(file_path: Path) -> Path:
    if file_path.suffix:
        stem = file_path.with_suffix("")
    else:
        stem = file_path
    return stem.parent / f"{stem.name}.embeddings.npz"


def _insight_source_signature(source: InsightSource) -> str:
    return json.dumps(source.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


def _append_unique_sources(
    existing: List[InsightSource],
    incoming: Iterable[InsightSource],
) -> None:
    seen = {_insight_source_signature(source) for source in existing}
    for source in incoming:
        signature = _insight_source_signature(source)
        if signature in seen:
            continue
        existing.append(source)
        seen.add(signature)


def _serialize_sources(sources: Iterable[InsightSource]) -> list[dict[str, Any]]:
    return [source.to_dict() for source in sources]


def _deserialize_sources(raw_sources: Any) -> list[InsightSource]:
    if not isinstance(raw_sources, list):
        return []
    deduped_sources: list[InsightSource] = []
    _append_unique_sources(
        deduped_sources,
        [
            coerce_insight_source(item)
            for item in raw_sources
            if isinstance(item, (InsightSource, dict))
        ],
    )
    return deduped_sources


# ---------------------------------------------------------------------------
# Update operations
# ---------------------------------------------------------------------------


@dataclass
class UpdateOperation:
    """Single mutation to apply to the skillbook."""

    type: OperationType
    section: str
    issue: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    insight: Optional[str] = None
    skill_id: Optional[str] = None
    metadata: Dict[str, int] = field(default_factory=dict)
    reason: Optional[str] = None
    insight_source: Optional[InsightSourceInput] = None
    learning_index: Optional[int] = None
    reflection_index: Optional[int] = None
    reflection_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "UpdateOperation":
        metadata_raw = payload.get("metadata") or {}
        metadata: Dict[str, Any] = (
            cast(Dict[str, Any], metadata_raw) if isinstance(metadata_raw, dict) else {}
        )

        op_type = str(payload["type"]).upper()
        if op_type not in ("ADD", "UPDATE", "TAG", "REMOVE"):
            raise ValueError(f"Invalid operation type: {op_type}")

        raw_source = payload.get("insight_source")
        insight_source: Optional[InsightSourceInput] = None
        if isinstance(raw_source, dict):
            insight_source = InsightSource.from_dict(cast(Dict[str, Any], raw_source))
        elif isinstance(raw_source, Iterable) and not isinstance(
            raw_source, (str, bytes)
        ):
            insight_source = [
                InsightSource.from_dict(cast(Dict[str, Any], item))
                for item in raw_source
                if isinstance(item, dict)
            ]

        raw_learning_index = payload.get("learning_index")
        learning_index: Optional[int] = None
        if raw_learning_index is not None:
            try:
                learning_index = int(cast(int, raw_learning_index))
            except (TypeError, ValueError):
                pass

        raw_reflection_index = payload.get("reflection_index")
        reflection_index: Optional[int] = None
        if raw_reflection_index is not None:
            try:
                reflection_index = int(cast(int, raw_reflection_index))
            except (TypeError, ValueError):
                pass

        reflection_indices: List[int] = []
        raw_reflection_indices = payload.get("reflection_indices")
        if isinstance(raw_reflection_indices, Iterable) and not isinstance(
            raw_reflection_indices, (str, bytes)
        ):
            for value in raw_reflection_indices:
                try:
                    reflection_indices.append(int(cast(int, value)))
                except (TypeError, ValueError):
                    continue

        raw_keywords = payload.get("keywords")
        keywords: list[str] = []
        if isinstance(raw_keywords, Iterable) and not isinstance(
            raw_keywords, (str, bytes)
        ):
            keywords = _normalize_keywords(raw_keywords)

        issue = _normalize_optional_text(payload.get("issue"))
        insight = _normalize_optional_text(payload.get("insight"))
        reason = _normalize_optional_text(payload.get("reason"))

        return cls(
            type=cast(OperationType, op_type),
            section=str(payload.get("section", "")),
            issue=issue,
            keywords=keywords,
            insight=insight,
            skill_id=(
                str(payload["skill_id"])
                if payload.get("skill_id") is not None
                else None
            ),
            metadata={str(k): int(v) for k, v in metadata.items()},
            reason=reason,
            insight_source=insight_source,
            learning_index=learning_index,
            reflection_index=reflection_index,
            reflection_indices=reflection_indices,
        )

    def to_json(self) -> Dict[str, object]:
        data: Dict[str, object] = {"type": self.type, "section": self.section}
        if self.issue is not None:
            data["issue"] = self.issue
        if self.keywords:
            data["keywords"] = list(self.keywords)
        if self.insight is not None:
            data["insight"] = self.insight
        if self.skill_id is not None:
            data["skill_id"] = self.skill_id
        if self.metadata:
            data["metadata"] = self.metadata
        if self.reason is not None:
            data["reason"] = self.reason
        if self.insight_source is not None:
            sources = coerce_insight_sources(self.insight_source)
            if len(sources) == 1:
                data["insight_source"] = sources[0].to_dict()
            elif sources:
                data["insight_source"] = [source.to_dict() for source in sources]
        if self.learning_index is not None:
            data["learning_index"] = self.learning_index
        if self.reflection_index is not None:
            data["reflection_index"] = self.reflection_index
        if self.reflection_indices:
            data["reflection_indices"] = list(self.reflection_indices)
        return data


@dataclass
class UpdateBatch:
    """Bundle of skill manager reasoning and operations."""

    reasoning: str
    operations: List[UpdateOperation] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "UpdateBatch":
        ops_payload = payload.get("operations")
        operations = []
        if isinstance(ops_payload, Iterable):
            for item in ops_payload:
                if isinstance(item, dict):
                    operations.append(UpdateOperation.from_json(item))
        return cls(reasoning=str(payload.get("reasoning", "")), operations=operations)

    def to_json(self) -> Dict[str, object]:
        return {
            "reasoning": self.reasoning,
            "operations": [op.to_json() for op in self.operations],
        }


# ---------------------------------------------------------------------------
# Skill types
# ---------------------------------------------------------------------------


@dataclass
class SimilarityDecision:
    """Record of a SkillManager decision to KEEP two skills separate."""

    decision: Literal["KEEP"]
    reasoning: str
    decided_at: str
    similarity_at_decision: float


@dataclass
class Skill:
    """Single skillbook entry."""

    id: str
    section: SkillSection
    keywords: list[str]
    issue: str
    insight: str | None = None
    occurrences: List[InsightSource] = field(default_factory=list)
    active: bool = True
    used_count: int = 0
    helpful_count: int = 0
    harmful_count: int = 0
    neutral_count: int = 0
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_llm_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "section": self.section,
            "keywords": list(self.keywords),
            "issue": self.issue,
            "insight": self.insight,
            "active": self.active,
            "used_count": self.used_count,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "neutral_count": self.neutral_count,
        }

    def embedding_text(self) -> str:
        parts = [self.issue]
        if self.insight:
            parts.append(self.insight)
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Skillbook
# ---------------------------------------------------------------------------


class Skillbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0
        self._similarity_decisions: Dict[FrozenSet[str], SimilarityDecision] = {}
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"Skillbook(skills={len(self._skills)}, sections={list(self._sections.keys())})"

    def __str__(self) -> str:
        if not self._skills:
            return "Skillbook(empty)"
        return self.as_prompt()

    @property
    def lock(self) -> threading.RLock:
        """Reentrant lock guarding all mutations."""
        return self._lock

    def clone(self) -> "Skillbook":
        """In-memory deep copy that preserves computed embeddings."""

        with self._lock:
            clone = Skillbook()
            clone._skills = {
                skill_id: copy.deepcopy(skill)
                for skill_id, skill in self._skills.items()
            }
            clone._sections = {
                section: list(skill_ids)
                for section, skill_ids in self._sections.items()
            }
            clone._next_id = self._next_id
            clone._similarity_decisions = dict(self._similarity_decisions)
            return clone

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def add_skill(
        self,
        section: str,
        issue: str | None = None,
        *,
        keywords: Iterable[Any] | None = None,
        insight: str | None = None,
        skill_id: Optional[str] = None,
        insight_source: Optional[InsightSourceInput] = None,
    ) -> Skill:
        with self._lock:
            raw_section = _normalize_required_text(section, "section").lower()
            normalized_section, normalized_keywords = _coerce_section_and_keywords(
                section, keywords
            )
            issue_text = _normalize_required_text(issue, "issue")
            insight_text = _normalize_optional_text(insight)

            if not normalized_keywords:
                normalized_keywords = _normalize_keywords([raw_section])
            if normalized_section == "context" and insight_text is None:
                if raw_section not in VALID_SECTIONS:
                    insight_text = issue_text
                else:
                    raise ValueError("context skills require a non-empty insight")

            skill_id = skill_id or self._generate_id(normalized_section)
            skill = Skill(
                id=skill_id,
                section=normalized_section,
                keywords=normalized_keywords,
                issue=issue_text,
                insight=insight_text,
            )
            _append_unique_sources(
                skill.occurrences, coerce_insight_sources(insight_source)
            )
            self._skills[skill_id] = skill
            self._sections.setdefault(normalized_section, []).append(skill_id)
            return skill

    def update_skill(
        self,
        skill_id: str,
        *,
        issue: object = _UNSET,
        keywords: object = _UNSET,
        insight: object = _UNSET,
        insight_source: Optional[InsightSourceInput] = None,
    ) -> Optional[Skill]:
        with self._lock:
            skill = self._skills.get(skill_id)
            if skill is None:
                return None

            if issue is not _UNSET and issue is not None:
                skill.issue = _normalize_required_text(issue, "issue")

            if keywords is not _UNSET and keywords is not None:
                normalized_keywords = _normalize_keywords(cast(Iterable[Any], keywords))
                if not normalized_keywords:
                    raise ValueError("keywords are required and must be non-empty")
                skill.keywords = normalized_keywords

            if insight is not _UNSET and insight is not None:
                skill.insight = _normalize_optional_text(insight)

            if skill.section == "context" and not skill.insight:
                raise ValueError("context skills require a non-empty insight")

            if insight_source is not None:
                _append_unique_sources(
                    skill.occurrences,
                    coerce_insight_sources(insight_source),
                )

            skill.embedding = None
            skill.updated_at = _now_iso()
            return skill

    def tag_skill(
        self,
        skill_id: str,
        delta: Literal[1, -1, 0],
        *,
        insight_source: Optional[InsightSourceInput] = None,
    ) -> Optional[Skill]:
        """Record an effectiveness observation for a skill."""
        with self._lock:
            skill = self._skills.get(skill_id)
            if skill is None:
                return None
            if delta == 1:
                skill.helpful_count += 1
            elif delta == -1:
                skill.harmful_count += 1
            else:
                skill.neutral_count += 1
            if insight_source is not None:
                _append_unique_sources(
                    skill.occurrences,
                    coerce_insight_sources(insight_source),
                )
            skill.updated_at = _now_iso()
            return skill

    def mark_used(self, skill_ids: Iterable[str]) -> None:
        """Bump ``used_count`` for each active skill ID."""
        with self._lock:
            for sid in skill_ids:
                skill = self._skills.get(sid)
                if skill is not None and skill.active:
                    skill.used_count += 1
                    skill.updated_at = _now_iso()

    def remove_skill(
        self,
        skill_id: str,
        soft: bool = True,
        *,
        insight_source: Optional[InsightSourceInput] = None,
    ) -> None:
        with self._lock:
            skill = self._skills.get(skill_id)
            if skill is None:
                return
            if soft:
                skill.active = False
                if insight_source is not None:
                    _append_unique_sources(
                        skill.occurrences,
                        coerce_insight_sources(insight_source),
                    )
                skill.updated_at = _now_iso()
            else:
                self.purge(skill_id)

    def purge(self, skill_id: str) -> None:
        with self._lock:
            skill = self._skills.pop(skill_id, None)
            if skill is None:
                return
            section_list = self._sections.get(skill.section)
            if section_list:
                self._sections[skill.section] = [
                    sid for sid in section_list if sid != skill_id
                ]
                if not self._sections[skill.section]:
                    del self._sections[skill.section]

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def skills(self, include_invalid: bool = False) -> List[Skill]:
        if include_invalid:
            return list(self._skills.values())
        return [s for s in self._skills.values() if s.active]

    # ------------------------------------------------------------------ #
    # Similarity decisions
    # ------------------------------------------------------------------ #

    def get_similarity_decision(
        self, skill_id_a: str, skill_id_b: str
    ) -> Optional[SimilarityDecision]:
        pair_key = frozenset([skill_id_a, skill_id_b])
        return self._similarity_decisions.get(pair_key)

    def set_similarity_decision(
        self,
        skill_id_a: str,
        skill_id_b: str,
        decision: SimilarityDecision,
    ) -> None:
        with self._lock:
            pair_key = frozenset([skill_id_a, skill_id_b])
            self._similarity_decisions[pair_key] = decision

    def has_keep_decision(self, skill_id_a: str, skill_id_b: str) -> bool:
        decision = self.get_similarity_decision(skill_id_a, skill_id_b)
        return decision is not None and decision.decision == "KEEP"

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self, exclude_embeddings: bool = False) -> Dict[str, object]:
        del exclude_embeddings  # JSON never carries embeddings in v2.
        similarity_decisions_serialized = {
            ",".join(sorted(pair_ids)): asdict(decision)
            for pair_ids, decision in self._similarity_decisions.items()
        }
        skills_serialized = {}
        for skill_id, skill in self._skills.items():
            skill_dict = asdict(skill)
            skill_dict.pop("embedding", None)
            skill_dict["occurrences"] = _serialize_sources(skill.occurrences)
            skills_serialized[skill_id] = skill_dict
        return {
            "schema_version": SCHEMA_VERSION,
            "skills": skills_serialized,
            "sections": self._sections,
            "next_id": self._next_id,
            "similarity_decisions": similarity_decisions_serialized,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Skillbook":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != SCHEMA_VERSION:
            raise ValueError("Skillbook format v2 required — regenerate")

        instance = cls()
        skills_payload = payload.get("skills", {})
        if isinstance(skills_payload, dict):
            for skill_id, skill_value in skills_payload.items():
                if not isinstance(skill_value, dict):
                    continue
                skill_data = dict(skill_value)
                skill_data["embedding"] = None
                skill_data["active"] = bool(skill_data.get("active", True))
                raw_keywords = skill_data.get("keywords")
                skill_data["keywords"] = _normalize_keywords(
                    raw_keywords if isinstance(raw_keywords, list) else []
                )
                if not skill_data["keywords"]:
                    legacy_section = skill_data.get("section", DEFAULT_LEGACY_SECTION)
                    skill_data["keywords"] = _normalize_keywords([legacy_section])

                section_value = str(skill_data.get("section", DEFAULT_LEGACY_SECTION))
                normalized_section, normalized_keywords = _coerce_section_and_keywords(
                    section_value,
                    skill_data["keywords"],
                )
                skill_data["section"] = normalized_section
                skill_data["keywords"] = normalized_keywords
                skill_data["issue"] = _normalize_required_text(
                    skill_data.get("issue"), "issue"
                )
                skill_data["insight"] = _normalize_optional_text(
                    skill_data.get("insight")
                )
                skill_data["occurrences"] = _deserialize_sources(
                    skill_data.get("occurrences")
                )
                valid_fields = {f.name for f in dataclass_fields(Skill)}
                skill_data = {k: v for k, v in skill_data.items() if k in valid_fields}
                instance._skills[str(skill_id)] = Skill(**skill_data)

        sections_payload = payload.get("sections", {})
        if isinstance(sections_payload, dict):
            normalized_sections: dict[str, list[str]] = {}
            for section, ids in sections_payload.items():
                if not isinstance(ids, Iterable) or isinstance(ids, (str, bytes)):
                    continue
                normalized_sections[str(section)] = [str(item) for item in ids]
            instance._sections = normalized_sections
        next_id_value = payload.get("next_id", 0)
        instance._next_id = (
            int(cast(Union[int, str], next_id_value))
            if next_id_value is not None
            else 0
        )
        similarity_decisions_payload = payload.get("similarity_decisions", {})
        if isinstance(similarity_decisions_payload, dict):
            for pair_key_str, decision_value in similarity_decisions_payload.items():
                if isinstance(decision_value, dict):
                    pair_ids = frozenset(str(pair_key_str).split(","))
                    instance._similarity_decisions[pair_ids] = SimilarityDecision(
                        **decision_value
                    )

        # Prefer the explicit serialized section ordering, but rebuild if needed.
        if not instance._sections:
            for skill in instance._skills.values():
                instance._sections.setdefault(skill.section, []).append(skill.id)

        return instance

    def dumps(self, exclude_embeddings: bool = False) -> str:
        return json.dumps(
            self.to_dict(exclude_embeddings=exclude_embeddings),
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def loads(cls, data: str) -> "Skillbook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Skillbook serialization must be a JSON object.")
        return cls.from_dict(payload)

    def save_to_file(self, path: str, exclude_embeddings: bool = False) -> None:
        file_path = Path(path)
        sidecar_path = _embedding_sidecar_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.dumps(exclude_embeddings=True))

        if exclude_embeddings:
            return

        embeddings = {
            skill.id: skill.embedding
            for skill in self._skills.values()
            if skill.embedding is not None
        }
        if not embeddings:
            if sidecar_path.exists():
                sidecar_path.unlink()
            return

        import numpy as np

        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            skill_id: np.asarray(embedding, dtype="float32")
            for skill_id, embedding in embeddings.items()
        }
        np.savez_compressed(sidecar_path, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load_from_file(cls, path: str) -> "Skillbook":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Skillbook file not found: {path}")
        with file_path.open("r", encoding="utf-8") as f:
            skillbook = cls.loads(f.read())

        sidecar_path = _embedding_sidecar_path(file_path)
        if not sidecar_path.exists():
            return skillbook

        import numpy as np

        with np.load(sidecar_path) as embeddings:
            for skill_id in embeddings.files:
                skill = skillbook.get_skill(skill_id)
                if skill is not None:
                    skill.embedding = embeddings[skill_id].astype("float32").tolist()
        return skillbook

    # ------------------------------------------------------------------ #
    # Update application
    # ------------------------------------------------------------------ #

    def apply_update(self, update: UpdateBatch) -> None:
        with self._lock:
            for operation in update.operations:
                self._apply_operation(operation)

    def _apply_operation(self, operation: UpdateOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            self.add_skill(
                section=operation.section,
                issue=operation.issue or "",
                keywords=operation.keywords,
                insight=operation.insight,
                skill_id=operation.skill_id,
                insight_source=operation.insight_source,
            )
        elif op_type == "UPDATE":
            if operation.skill_id is None:
                return
            self.update_skill(
                operation.skill_id,
                issue=operation.issue if operation.issue is not None else _UNSET,
                keywords=operation.keywords if operation.keywords else _UNSET,
                insight=operation.insight if operation.insight is not None else _UNSET,
                insight_source=operation.insight_source,
            )
        elif op_type == "TAG":
            if operation.skill_id is None:
                return
            delta = int(operation.metadata.get("delta", 0))
            if delta > 0:
                delta = 1
            elif delta < 0:
                delta = -1
            self.tag_skill(
                operation.skill_id,
                cast(Literal[1, -1, 0], delta),
                insight_source=operation.insight_source,
            )
        elif op_type == "REMOVE":
            if operation.skill_id is None:
                return
            self.remove_skill(
                operation.skill_id, insight_source=operation.insight_source
            )

    # ------------------------------------------------------------------ #
    # Presentation
    # ------------------------------------------------------------------ #

    def as_prompt(self) -> str:
        parts: List[str] = []
        for section in ("context", "harness"):
            skill_ids = self._sections.get(section, [])
            section_skills = [
                self._skills[sid] for sid in skill_ids if self._skills[sid].active
            ]
            if not section_skills:
                continue
            parts.append(f"## {section}")
            for skill in section_skills:
                parts.append(f"- [{skill.id}]")
                parts.append(f"  Keywords: {', '.join(skill.keywords)}")
                parts.append(f"  Issue: {skill.issue}")
                if skill.insight:
                    parts.append(f"  Insight: {skill.insight}")
                parts.append("")
        return "\n".join(parts).rstrip()

    def stats(self) -> Dict[str, object]:
        active_skills = [skill for skill in self._skills.values() if skill.active]
        by_section: dict[str, int] = {section: 0 for section in VALID_SECTIONS}
        for skill in active_skills:
            by_section[skill.section] = by_section.get(skill.section, 0) + 1
        return {
            "sections": len(
                [section for section, count in by_section.items() if count]
            ),
            "skills": len(self._skills),
            "active_skills": len(active_skills),
            "by_section": by_section,
        }

    # ------------------------------------------------------------------ #
    # Insight source analysis
    # ------------------------------------------------------------------ #

    def source_map(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            result: Dict[str, List[Dict[str, Any]]] = {}
            for skill_id, skill in self._skills.items():
                if skill.occurrences:
                    result[skill_id] = _serialize_sources(skill.occurrences)
            return result

    def source_summary(self) -> Dict[str, Any]:
        with self._lock:
            epochs: Dict[Optional[int], int] = {}
            source_systems: Dict[str, int] = {}
            trace_uids: Dict[str, int] = {}
            sample_questions: Dict[str, int] = {}
            total = 0
            for skill in self._skills.values():
                for src in skill.occurrences:
                    total += 1
                    epochs[src.epoch] = epochs.get(src.epoch, 0) + 1
                    source_systems[src.source_system] = (
                        source_systems.get(src.source_system, 0) + 1
                    )
                    trace_uids[src.trace_uid] = trace_uids.get(src.trace_uid, 0) + 1
                    sq = src.sample_question or ""
                    if sq:
                        sample_questions[sq] = sample_questions.get(sq, 0) + 1
            return {
                "total_sources": total,
                "epochs": epochs,
                "source_systems": source_systems,
                "trace_uids": trace_uids,
                "sample_questions": sample_questions,
            }

    def source_filter(
        self,
        *,
        epoch: Optional[int] = None,
        sample_question: Optional[str] = None,
        trace_uid: Optional[str] = None,
        trace_id: Optional[str] = None,
        source_system: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            result: Dict[str, List[Dict[str, Any]]] = {}
            for skill_id, skill in self._skills.items():
                matches = []
                for src in skill.occurrences:
                    if epoch is not None and src.epoch != epoch:
                        continue
                    if trace_uid is not None and src.trace_uid != trace_uid:
                        continue
                    if trace_id is not None and src.trace_id != trace_id:
                        continue
                    if source_system is not None and src.source_system != source_system:
                        continue
                    if sample_question is not None:
                        sq = src.sample_question or ""
                        if sample_question.lower() not in sq.lower():
                            continue
                    matches.append(src.to_dict())
                if matches:
                    result[skill_id] = matches
            return result

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _generate_id(self, section: SkillSection) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"
