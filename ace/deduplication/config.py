"""Configuration for skill deduplication."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DeduplicationConfig:
    """Configuration for skill deduplication.

    Attributes:
        enabled: Whether deduplication is enabled (default: True)
        embedding_model: Model to use for computing embeddings
        embedding_provider: Provider for embeddings ('litellm' or 'sentence_transformers')
        similarity_threshold: Minimum similarity score to consider skills as similar
        min_pairs_to_report: Minimum number of similar pairs before including in SkillManager prompt
        within_section_only: If True, only compare skills within the same section
    """

    # Feature flags
    enabled: bool = True

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["litellm", "sentence_transformers"] = "litellm"

    # Similarity thresholds
    similarity_threshold: float = 0.85

    # Cost control: only report similar pairs if >= this many found
    min_pairs_to_report: int = 1

    # Scope
    within_section_only: bool = True

    # Optional: sentence-transformers model (used if embedding_provider='sentence_transformers')
    local_model_name: str = "all-MiniLM-L6-v2"
