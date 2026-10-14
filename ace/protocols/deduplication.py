"""Protocol and config for skill deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from ..core.skillbook import Skillbook


@dataclass
class DeduplicationConfig:
    """Configuration for skill deduplication.

    ``consolidation_model`` is the model that decides what to do with each
    similar pair (merge, delete, keep or update). Without it the
    ``DeduplicateStep`` only logs the similarity report and leaves the
    skillbook unchanged. Accepts a model string or a PydanticAI ``Model``.
    """

    enabled: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["litellm", "sentence_transformers"] = "litellm"
    similarity_threshold: float = 0.85
    min_pairs_to_report: int = 1
    within_section_only: bool = True
    local_model_name: str = "all-MiniLM-L6-v2"
    consolidation_model: str | Model | None = None


@runtime_checkable
class DeduplicationManagerLike(Protocol):
    """Structural interface for deduplication managers.

    The concrete ``ace.deduplication.DeduplicationManager`` satisfies this.
    """

    def get_similarity_report(self, skillbook: "Skillbook") -> Optional[str]: ...
