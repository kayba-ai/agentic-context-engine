"""Skillbook storage and mutation logic."""

from __future__ import annotations

import json
from dataclasses import asdict, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Union, cast

from .skill import Skill, SimilarityDecision
from .updates import UpdateBatch, UpdateOperation


class Skillbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0
        self._similarity_decisions: Dict[FrozenSet[str], SimilarityDecision] = {}

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
        metadata: Optional[Dict[str, int]] = None,
        justification: Optional[str] = None,
        evidence: Optional[str] = None,
        insight_source: Optional[Dict[str, Any]] = None,
    ) -> Skill:
        skill_id = skill_id or self._generate_id(section)
        metadata = metadata or {}
        skill = Skill(
            id=skill_id,
            section=section,
            content=content,
            justification=justification,
            evidence=evidence,
            sources=[insight_source] if insight_source else [],
        )
        skill.apply_metadata(metadata)
        self._skills[skill_id] = skill
        self._sections.setdefault(section, []).append(skill_id)
        return skill

    def update_skill(
        self,
        skill_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
        justification: Optional[str] = None,
        evidence: Optional[str] = None,
        insight_source: Optional[Dict[str, Any]] = None,
    ) -> Optional[Skill]:
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        if content is not None:
            skill.content = content
        if justification is not None:
            skill.justification = justification
        if evidence is not None:
            skill.evidence = evidence
        if metadata:
            skill.apply_metadata(metadata)
        if insight_source is not None:
            skill.sources.append(insight_source)
        skill.updated_at = datetime.now(timezone.utc).isoformat()
        return skill

    def tag_skill(
        self, skill_id: str, tag: str, increment: int = 1
    ) -> Optional[Skill]:
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        skill.tag(tag, increment=increment)
        return skill

    def remove_skill(self, skill_id: str, soft: bool = False) -> None:
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
            if exclude_embeddings:
                skill_dict["embedding"] = None
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
                    if "sources" not in skill_data:
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
        for operation in update.operations:
            self._apply_operation(operation)

    def _apply_operation(self, operation: UpdateOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            self.add_skill(
                section=operation.section,
                content=operation.content or "",
                skill_id=operation.skill_id,
                metadata=operation.metadata,
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
                metadata=operation.metadata,
                justification=operation.justification,
                evidence=operation.evidence,
                insight_source=operation.insight_source,
            )
        elif op_type == "TAG":
            if operation.skill_id is None:
                return
            valid_tags = {"helpful", "harmful", "neutral"}
            for tag, increment in operation.metadata.items():
                if tag in valid_tags:
                    self.tag_skill(operation.skill_id, tag, increment)
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
                counters = f"(helpful={skill.helpful}, harmful={skill.harmful}, neutral={skill.neutral})"
                parts.append(f"- [{skill.id}] {skill.content} {counters}")
        return "\n".join(parts)

    def stats(self) -> Dict[str, object]:
        return {
            "sections": len(self._sections),
            "skills": len(self._skills),
            "tags": {
                "helpful": sum(s.helpful for s in self._skills.values()),
                "harmful": sum(s.harmful for s in self._skills.values()),
                "neutral": sum(s.neutral for s in self._skills.values()),
            },
        }

    # ------------------------------------------------------------------ #
    # Insight source analysis
    # ------------------------------------------------------------------ #

    def source_map(self) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for skill_id, skill in self._skills.items():
            if skill.sources:
                result[skill_id] = list(skill.sources)
        return result

    def source_summary(self) -> Dict[str, Any]:
        epochs: Dict[int, int] = {}
        sample_questions: Dict[str, int] = {}
        total = 0
        for skill in self._skills.values():
            for src in skill.sources:
                total += 1
                ep = src.get("epoch", 0)
                epochs[ep] = epochs.get(ep, 0) + 1
                sq = src.get("sample_question", "")
                if sq:
                    sample_questions[sq] = sample_questions.get(sq, 0) + 1
        return {
            "total_sources": total,
            "epochs": epochs,
            "sample_questions": sample_questions,
        }

    def source_filter(
        self,
        *,
        epoch: Optional[int] = None,
        sample_question: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for skill_id, skill in self._skills.items():
            matches = []
            for src in skill.sources:
                if epoch is not None and src.get("epoch") != epoch:
                    continue
                if sample_question is not None:
                    sq = src.get("sample_question", "")
                    if sample_question.lower() not in sq.lower():
                        continue
                matches.append(src)
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
