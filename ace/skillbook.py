"""Skillbook storage and mutation logic for ACE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Optional, Union, cast

from .updates import UpdateBatch, UpdateOperation


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
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Deduplication fields
    embedding: Optional[List[float]] = None
    status: Literal["active", "invalid"] = "active"

    def apply_metadata(self, metadata: Dict[str, int]) -> None:
        for key, value in metadata.items():
            if hasattr(self, key):
                setattr(self, key, int(value))

    def tag(self, tag: str, increment: int = 1) -> None:
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_llm_dict(self) -> Dict[str, Any]:
        """
        Return dictionary with only LLM-relevant fields.

        Excludes created_at and updated_at which are internal metadata
        not useful for LLM strategy selection.

        Returns:
            Dict with id, section, content, helpful, harmful, neutral
        """
        return {
            "id": self.id,
            "section": self.section,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "neutral": self.neutral,
        }


class Skillbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0
        # Store KEEP decisions so we don't re-ask about the same pairs
        self._similarity_decisions: Dict[FrozenSet[str], SimilarityDecision] = {}

    def __repr__(self) -> str:
        """Concise representation for debugging and object inspection."""
        return f"Skillbook(skills={len(self._skills)}, sections={list(self._sections.keys())})"

    def __str__(self) -> str:
        """
        Human-readable representation showing actual skillbook content.

        Uses markdown format for readability (not TOON) since this is
        typically used for debugging/inspection, not LLM prompts.
        """
        if not self._skills:
            return "Skillbook(empty)"
        return self._as_markdown_debug()

    # ------------------------------------------------------------------ #
    # CRUD utils
    # ------------------------------------------------------------------ #
    def add_skill(
        self,
        section: str,
        content: str,
        skill_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
        justification: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> Skill:
        skill_id = skill_id or self._generate_id(section)
        metadata = metadata or {}
        skill = Skill(
            id=skill_id,
            section=section,
            content=content,
            justification=justification,
            evidence=evidence,
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
        skill.updated_at = datetime.now(timezone.utc).isoformat()
        return skill

    def tag_skill(self, skill_id: str, tag: str, increment: int = 1) -> Optional[Skill]:
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        skill.tag(tag, increment=increment)

        # Opik tracing handles this automatically via @track decorator

        return skill

    def remove_skill(self, skill_id: str, soft: bool = False) -> None:
        """Remove a skill from the skillbook.

        Args:
            skill_id: ID of the skill to remove
            soft: If True, mark as invalid instead of deleting (for audit trail)
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            return

        if soft:
            # Soft delete: mark as invalid but keep in storage
            skill.status = "invalid"
            skill.updated_at = datetime.now(timezone.utc).isoformat()
        else:
            # Hard delete: remove entirely
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
        """Get all skills in the skillbook.

        Args:
            include_invalid: If True, include soft-deleted skills

        Returns:
            List of skills (active only by default)
        """
        if include_invalid:
            return list(self._skills.values())
        return [s for s in self._skills.values() if s.status == "active"]

    # ------------------------------------------------------------------ #
    # Similarity decisions (for deduplication)
    # ------------------------------------------------------------------ #
    def get_similarity_decision(
        self, skill_id_a: str, skill_id_b: str
    ) -> Optional[SimilarityDecision]:
        """Get a prior similarity decision for a pair of skills."""
        pair_key = frozenset([skill_id_a, skill_id_b])
        return self._similarity_decisions.get(pair_key)

    def set_similarity_decision(
        self,
        skill_id_a: str,
        skill_id_b: str,
        decision: SimilarityDecision,
    ) -> None:
        """Store a similarity decision for a pair of skills."""
        pair_key = frozenset([skill_id_a, skill_id_b])
        self._similarity_decisions[pair_key] = decision

    def has_keep_decision(self, skill_id_a: str, skill_id_b: str) -> bool:
        """Check if there's a KEEP decision for this pair."""
        decision = self.get_similarity_decision(skill_id_a, skill_id_b)
        return decision is not None and decision.decision == "KEEP"

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self, exclude_embeddings: bool = False) -> Dict[str, object]:
        # Serialize similarity decisions with string keys (JSON doesn't support frozenset)
        similarity_decisions_serialized = {
            ",".join(sorted(pair_ids)): asdict(decision)
            for pair_ids, decision in self._similarity_decisions.items()
        }
        # Serialize skills, optionally excluding embeddings
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
                    # Handle new optional fields with defaults for backwards compatibility
                    skill_data = dict(skill_value)
                    if "embedding" not in skill_data:
                        skill_data["embedding"] = None
                    if "status" not in skill_data:
                        skill_data["status"] = "active"
                    if "justification" not in skill_data:
                        skill_data["justification"] = None
                    if "evidence" not in skill_data:
                        skill_data["evidence"] = None
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
        # Deserialize similarity decisions
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
        """Save skillbook to a JSON file.

        Args:
            path: File path where to save the skillbook
            exclude_embeddings: If True, set embeddings to None in the output.
                Useful for smaller files and version control. Default False.

        Example:
            >>> skillbook.save_to_file("trained_model.json")
            >>> skillbook.save_to_file("skillbook_light.json", exclude_embeddings=True)
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.dumps(exclude_embeddings=exclude_embeddings))

    @classmethod
    def load_from_file(cls, path: str) -> "Skillbook":
        """Load skillbook from a JSON file.

        Args:
            path: File path to load the skillbook from

        Returns:
            Skillbook instance loaded from the file

        Example:
            >>> skillbook = Skillbook.load_from_file("trained_model.json")

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the JSON doesn't represent a valid skillbook
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Skillbook file not found: {path}")
        with file_path.open("r", encoding="utf-8") as f:
            return cls.loads(f.read())

    # ------------------------------------------------------------------ #
    # Update application
    # ------------------------------------------------------------------ #
    def apply_update(self, update: UpdateBatch) -> None:
        skills_before = len(self._skills)

        for operation in update.operations:
            self._apply_operation(operation)

        skills_after = len(self._skills)

        # Opik tracing handles this automatically via @track decorator

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
            )
        elif op_type == "TAG":
            if operation.skill_id is None:
                return
            # Only apply valid tag names as defensive measure
            valid_tags = {"helpful", "harmful", "neutral"}
            for tag, increment in operation.metadata.items():
                if tag in valid_tags:
                    self.tag_skill(operation.skill_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.skill_id is None:
                return
            self.remove_skill(operation.skill_id)

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def as_prompt(self) -> str:
        """
        Return TOON-encoded skillbook for LLM prompts.

        Uses tab delimiters and excludes internal metadata (created_at, updated_at)
        for maximum token efficiency (~16-62% savings vs markdown).

        Returns:
            TOON-formatted string with skills array

        Raises:
            ImportError: If python-toon is not installed
        """
        try:
            from toon import encode
        except ImportError:
            raise ImportError(
                "TOON compression requires python-toon. "
                "Install with: pip install python-toon>=0.1.0"
            )

        # Only include LLM-relevant fields (exclude created_at, updated_at)
        skills_data = [s.to_llm_dict() for s in self.skills()]

        # Use tab delimiter for 5-10% better compression than comma
        return encode({"skills": skills_data}, {"delimiter": "\t"})

    def _as_markdown_debug(self) -> str:
        """
        Human-readable markdown format for debugging/inspection only.

        This format is more readable than TOON but uses more tokens.
        Use for debugging, logging, or human inspection - not for LLM prompts.

        Returns:
            Markdown-formatted skillbook string
        """
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
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"
