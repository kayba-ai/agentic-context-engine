"""Skill dataclass and related types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


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
    embedding: Optional[List[float]] = None
    status: Literal["active", "invalid"] = "active"
    sources: List[Dict[str, Any]] = field(default_factory=list)

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
        return {
            "id": self.id,
            "section": self.section,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "neutral": self.neutral,
        }
