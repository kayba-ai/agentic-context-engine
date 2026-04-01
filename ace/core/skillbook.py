"""Skill, Skillbook, and update operations for the ACE framework."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

from .insight_source import (
    InsightSource,
    coerce_insight_source,
    coerce_insight_sources,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SKILL_TAGS: FrozenSet[str] = frozenset({"helpful", "harmful", "neutral"})  # deprecated

# ---------------------------------------------------------------------------
# Update operations
# ---------------------------------------------------------------------------

OperationType = Literal["ADD", "UPDATE", "TAG", "REMOVE"]
InsightSourceInput = Union[
    InsightSource,
    Dict[str, Any],
    List[Union[InsightSource, Dict[str, Any]]],
]


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


@dataclass
class UpdateOperation:
    """Single mutation to apply to the skillbook."""

    type: OperationType
    section: str
    content: Optional[str] = None
    skill_id: Optional[str] = None
    metadata: Dict[str, int] = field(default_factory=dict)
    justification: Optional[str] = None
    evidence: Optional[str] = None
    insight_source: Optional[InsightSourceInput] = None
    learning_index: Optional[int] = None
    reflection_index: Optional[int] = None
    reflection_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "UpdateOperation":
        # Filter metadata for TAG operations to only include valid tags
        metadata_raw = payload.get("metadata") or {}
        metadata: Dict[str, Any] = (
            cast(Dict[str, Any], metadata_raw) if isinstance(metadata_raw, dict) else {}
        )

        if str(payload["type"]).upper() == "TAG":
            metadata = {k: v for k, v in metadata.items() if str(k) in VALID_SKILL_TAGS}

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

        return cls(
            type=cast(OperationType, op_type),
            section=str(payload.get("section", "")),
            content=(
                str(payload["content"]) if payload.get("content") is not None else None
            ),
            skill_id=(
                str(payload["skill_id"])
                if payload.get("skill_id") is not None
                else None
            ),
            metadata={str(k): int(v) for k, v in metadata.items()},
            justification=(
                str(payload["justification"])
                if payload.get("justification") is not None
                else None
            ),
            evidence=(
                str(payload["evidence"])
                if payload.get("evidence") is not None
                else None
            ),
            insight_source=insight_source,
            learning_index=learning_index,
            reflection_index=reflection_index,
            reflection_indices=reflection_indices,
        )

    def to_json(self) -> Dict[str, object]:
        data: Dict[str, object] = {"type": self.type, "section": self.section}
        if self.content is not None:
            data["content"] = self.content
        if self.skill_id is not None:
            data["skill_id"] = self.skill_id
        if self.metadata:
            data["metadata"] = self.metadata
        if self.justification is not None:
            data["justification"] = self.justification
        if self.evidence is not None:
            data["evidence"] = self.evidence
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
    section: str
    content: str
    justification: Optional[str] = None
    evidence: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    embedding: Optional[List[float]] = None
    status: Literal["active", "invalid"] = "active"
    sources: List[InsightSource] = field(default_factory=list)

    def to_llm_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "section": self.section,
            "content": self.content,
        }


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
        return self._as_markdown_debug()

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def add_skill(
        self,
        section: str,
        content: str,
        skill_id: Optional[str] = None,
        justification: Optional[str] = None,
        evidence: Optional[str] = None,
        insight_source: Optional[InsightSourceInput] = None,
    ) -> Skill:
        with self._lock:
            skill_id = skill_id or self._generate_id(section)
            skill = Skill(
                id=skill_id,
                section=section,
                content=content,
                justification=justification,
                evidence=evidence,
                sources=[],
            )
            _append_unique_sources(
                skill.sources, coerce_insight_sources(insight_source)
            )
            self._skills[skill_id] = skill
            self._sections.setdefault(section, []).append(skill_id)
            return skill

    def update_skill(
        self,
        skill_id: str,
        *,
        content: Optional[str] = None,
        justification: Optional[str] = None,
        evidence: Optional[str] = None,
        insight_source: Optional[InsightSourceInput] = None,
    ) -> Optional[Skill]:
        with self._lock:
            skill = self._skills.get(skill_id)
            if skill is None:
                return None
            if content is not None:
                skill.content = content
            if justification is not None:
                skill.justification = justification
            if evidence is not None:
                skill.evidence = evidence
            if insight_source is not None:
                _append_unique_sources(
                    skill.sources,
                    coerce_insight_sources(insight_source),
                )
            skill.updated_at = datetime.now(timezone.utc).isoformat()
            return skill

    def remove_skill(self, skill_id: str, soft: bool = False) -> None:
        with self._lock:
            skill = self._skills.get(skill_id)
            if skill is None:
                return
            if soft:
                skill.status = "invalid"
                skill.updated_at = datetime.now(timezone.utc).isoformat()
            else:
                self._skills.pop(skill_id, None)
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
        return [s for s in self._skills.values() if s.status == "active"]

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
        similarity_decisions_serialized = {
            ",".join(sorted(pair_ids)): asdict(decision)
            for pair_ids, decision in self._similarity_decisions.items()
        }
        skills_serialized = {}
        for skill_id, skill in self._skills.items():
            skill_dict = asdict(skill)
            # Omit embedding when null or explicitly excluded
            if skill_dict.get("embedding") is None or exclude_embeddings:
                del skill_dict["embedding"]
            # Use InsightSource.to_dict() for clean sparse serialization
            # instead of the recursive asdict() which dumps empty/null fields.
            skill_dict["sources"] = [
                source.to_dict() for source in skill.sources
            ]
            skills_serialized[skill_id] = skill_dict
        return {
            "skills": skills_serialized,
            "sections": self._sections,
            "next_id": self._next_id,
            "similarity_decisions": similarity_decisions_serialized,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Skillbook":
        instance = cls()
        skills_payload = payload.get("skills", {})
        if isinstance(skills_payload, dict):
            for skill_id, skill_value in skills_payload.items():
                if isinstance(skill_value, dict):
                    skill_data = dict(skill_value)
                    if "embedding" not in skill_data:
                        skill_data["embedding"] = None
                    if "status" not in skill_data:
                        skill_data["status"] = "active"
                    if "justification" not in skill_data:
                        skill_data["justification"] = None
                    if "evidence" not in skill_data:
                        skill_data["evidence"] = None
                    raw_sources = skill_data.get("sources") or []
                    if isinstance(raw_sources, list):
                        deduped_sources: list[InsightSource] = []
                        _append_unique_sources(
                            deduped_sources,
                            [
                                coerce_insight_source(item)
                                for item in raw_sources
                                if isinstance(item, (InsightSource, dict))
                            ],
                        )
                        skill_data["sources"] = deduped_sources
                    else:
                        skill_data["sources"] = []
                    valid_fields = {f.name for f in dataclass_fields(Skill)}
                    skill_data = {
                        k: v for k, v in skill_data.items() if k in valid_fields
                    }
                    instance._skills[skill_id] = Skill(**skill_data)
        sections_payload = payload.get("sections", {})
        if isinstance(sections_payload, dict):
            instance._sections = {
                section: list(ids) if isinstance(ids, Iterable) else []
                for section, ids in sections_payload.items()
            }
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
                    pair_ids = frozenset(pair_key_str.split(","))
                    instance._similarity_decisions[pair_ids] = SimilarityDecision(
                        **decision_value
                    )
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
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.dumps(exclude_embeddings=exclude_embeddings))

    @classmethod
    def load_from_file(cls, path: str) -> "Skillbook":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Skillbook file not found: {path}")
        with file_path.open("r", encoding="utf-8") as f:
            return cls.loads(f.read())

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
                content=operation.content or "",
                skill_id=operation.skill_id,
                justification=operation.justification,
                evidence=operation.evidence,
                insight_source=operation.insight_source,
            )
        elif op_type == "UPDATE":
            if operation.skill_id is None:
                return
            self.update_skill(
                operation.skill_id,
                content=operation.content,
                justification=operation.justification,
                evidence=operation.evidence,
                insight_source=operation.insight_source,
            )
        elif op_type == "TAG":
            pass  # Tags are no longer tracked on skills
        elif op_type == "REMOVE":
            if operation.skill_id is None:
                return
            self.remove_skill(operation.skill_id)

    # ------------------------------------------------------------------ #
    # Presentation
    # ------------------------------------------------------------------ #

    def as_prompt(self) -> str:
        try:
            from toon import encode
        except ImportError:
            raise ImportError(
                "TOON compression requires python-toon. "
                "Install with: pip install python-toon>=0.1.0"
            )
        skills_data = [s.to_llm_dict() for s in self.skills()]
        return encode({"skills": skills_data}, {"delimiter": "\t"})

    def _as_markdown_debug(self) -> str:
        parts: List[str] = []
        for section, skill_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for skill_id in skill_ids:
                skill = self._skills[skill_id]
                parts.append(f"- [{skill.id}] {skill.content}")
        return "\n".join(parts)

    def stats(self) -> Dict[str, object]:
        return {
            "sections": len(self._sections),
            "skills": len(self._skills),
        }

    # ------------------------------------------------------------------ #
    # Insight source analysis
    # ------------------------------------------------------------------ #

    def source_map(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            result: Dict[str, List[Dict[str, Any]]] = {}
            for skill_id, skill in self._skills.items():
                if skill.sources:
                    result[skill_id] = [source.to_dict() for source in skill.sources]
            return result

    def source_summary(self) -> Dict[str, Any]:
        with self._lock:
            epochs: Dict[Optional[int], int] = {}
            source_systems: Dict[str, int] = {}
            trace_uids: Dict[str, int] = {}
            sample_questions: Dict[str, int] = {}
            total = 0
            for skill in self._skills.values():
                for src in skill.sources:
                    total += 1
                    epochs[src.epoch] = epochs.get(src.epoch, 0) + 1
                    source_systems[src.source_system] = (
                        source_systems.get(src.source_system, 0) + 1
                    )
                    trace_uids[src.trace_uid] = (
                        trace_uids.get(src.trace_uid, 0) + 1
                    )
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
                for src in skill.sources:
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

    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"
