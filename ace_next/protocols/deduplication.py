"""Protocol and config for skill deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..skillbook import Skillbook


@dataclass
class DeduplicationConfig:
    """Configuration for skill deduplication."""

    enabled: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["litellm", "sentence_transformers"] = "litellm"
    similarity_threshold: float = 0.85
    min_pairs_to_report: int = 1
    within_section_only: bool = True
    local_model_name: str = "all-MiniLM-L6-v2"


@runtime_checkable
class DeduplicationManagerLike(Protocol):
    """Structural interface for deduplication managers.

    The concrete ``ace.deduplication.DeduplicationManager`` satisfies this.
    """

    def get_similarity_report(self, skillbook: "Skillbook") -> Optional[str]: ...
