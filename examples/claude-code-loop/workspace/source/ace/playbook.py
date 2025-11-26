"""Playbook storage and mutation logic for ACE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, cast

from .delta import DeltaBatch, DeltaOperation


@dataclass
class Bullet:
    """Single playbook entry."""

    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

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


class Playbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._bullets: Dict[str, Bullet] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0

    def __repr__(self) -> str:
        """Concise representation for debugging and object inspection."""
        return f"Playbook(bullets={len(self._bullets)}, sections={list(self._sections.keys())})"

    def __str__(self) -> str:
        """
        Human-readable representation showing actual playbook content.

        Uses markdown format for readability (not TOON) since this is
        typically used for debugging/inspection, not LLM prompts.
        """
        if not self._bullets:
            return "Playbook(empty)"
        return self._as_markdown_debug()

    # ------------------------------------------------------------------ #
    # CRUD utils
    # ------------------------------------------------------------------ #
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Bullet:
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        bullet = Bullet(id=bullet_id, section=section, content=content)
        bullet.apply_metadata(metadata)
        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        return bullet

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        if content is not None:
            bullet.content = content
        if metadata:
            bullet.apply_metadata(metadata)
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        return bullet

    def tag_bullet(
        self, bullet_id: str, tag: str, increment: int = 1
    ) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.tag(tag, increment=increment)

        # Opik tracing handles this automatically via @track decorator

        return bullet

    def remove_bullet(self, bullet_id: str) -> None:
        bullet = self._bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section_list = self._sections.get(bullet.section)
        if section_list:
            self._sections[bullet.section] = [
                bid for bid in section_list if bid != bullet_id
            ]
            if not self._sections[bullet.section]:
                del self._sections[bullet.section]

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        return self._bullets.get(bullet_id)

    def bullets(self) -> List[Bullet]:
        return list(self._bullets.values())

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, object]:
        return {
            "bullets": {
                bullet_id: asdict(bullet) for bullet_id, bullet in self._bullets.items()
            },
            "sections": self._sections,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Playbook":
        instance = cls()
        bullets_payload = payload.get("bullets", {})
        if isinstance(bullets_payload, dict):
            for bullet_id, bullet_value in bullets_payload.items():
                if isinstance(bullet_value, dict):
                    instance._bullets[bullet_id] = Bullet(**bullet_value)
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
        return instance

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "Playbook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Playbook serialization must be a JSON object.")
        return cls.from_dict(payload)

    def save_to_file(self, path: str) -> None:
        """Save playbook to a JSON file.

        Args:
            path: File path where to save the playbook

        Example:
            >>> playbook.save_to_file("trained_model.json")
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.dumps())

    @classmethod
    def load_from_file(cls, path: str) -> "Playbook":
        """Load playbook from a JSON file.

        Args:
            path: File path to load the playbook from

        Returns:
            Playbook instance loaded from the file

        Example:
            >>> playbook = Playbook.load_from_file("trained_model.json")

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the JSON doesn't represent a valid playbook
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Playbook file not found: {path}")
        with file_path.open("r", encoding="utf-8") as f:
            return cls.loads(f.read())

    # ------------------------------------------------------------------ #
    # Delta application
    # ------------------------------------------------------------------ #
    def apply_delta(self, delta: DeltaBatch) -> None:
        bullets_before = len(self._bullets)

        for operation in delta.operations:
            self._apply_operation(operation)

        bullets_after = len(self._bullets)

        # Opik tracing handles this automatically via @track decorator

    def _apply_operation(self, operation: DeltaOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            self.add_bullet(
                section=operation.section,
                content=operation.content or "",
                bullet_id=operation.bullet_id,
                metadata=operation.metadata,
            )
        elif op_type == "UPDATE":
            if operation.bullet_id is None:
                return
            self.update_bullet(
                operation.bullet_id,
                content=operation.content,
                metadata=operation.metadata,
            )
        elif op_type == "TAG":
            if operation.bullet_id is None:
                return
            # Only apply valid tag names as defensive measure
            valid_tags = {"helpful", "harmful", "neutral"}
            for tag, increment in operation.metadata.items():
                if tag in valid_tags:
                    self.tag_bullet(operation.bullet_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.bullet_id is None:
                return
            self.remove_bullet(operation.bullet_id)

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def as_prompt(self) -> str:
        """
        Return TOON-encoded playbook for LLM prompts.

        Uses tab delimiters and excludes internal metadata (created_at, updated_at)
        for maximum token efficiency (~16-62% savings vs markdown).

        Returns:
            TOON-formatted string with bullets array

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
        bullets_data = [b.to_llm_dict() for b in self.bullets()]

        # Use tab delimiter for 5-10% better compression than comma
        return encode({"bullets": bullets_data}, {"delimiter": "\t"})

    def _as_markdown_debug(self) -> str:
        """
        Human-readable markdown format for debugging/inspection only.

        This format is more readable than TOON but uses more tokens.
        Use for debugging, logging, or human inspection - not for LLM prompts.

        Returns:
            Markdown-formatted playbook string
        """
        parts: List[str] = []
        for section, bullet_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self._bullets[bullet_id]
                counters = f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
                parts.append(f"- [{bullet.id}] {bullet.content} {counters}")
        return "\n".join(parts)

    def stats(self) -> Dict[str, object]:
        return {
            "sections": len(self._sections),
            "bullets": len(self._bullets),
            "tags": {
                "helpful": sum(b.helpful for b in self._bullets.values()),
                "harmful": sum(b.harmful for b in self._bullets.values()),
                "neutral": sum(b.neutral for b in self._bullets.values()),
            },
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"
