"""DeduplicateStep — periodically consolidates similar skills."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Protocol, runtime_checkable

from ..context import ACEStepContext
from ..skillbook import Skillbook

logger = logging.getLogger(__name__)


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
    """Protocol for deduplication managers.

    The concrete ``ace.deduplication.DeduplicationManager`` satisfies this
    structurally.  Steps depend on the protocol, not the implementation.
    """

    def get_similarity_report(self, skillbook: Skillbook) -> Optional[str]: ...


class DeduplicateStep:
    """Consolidate similar skills in the skillbook at a configurable interval.

    Optional side-effect step — appended to the pipeline by factory methods
    when ``dedup_config`` is provided.

    Stateless — uses ``ctx.global_sample_index`` with ``self.interval`` to
    skip most invocations.  Deduplication involves O(n^2) similarity
    comparisons, so running on every sample would be expensive.
    """

    requires = frozenset({"global_sample_index"})
    provides = frozenset()

    max_workers = 1

    def __init__(
        self,
        manager: DeduplicationManagerLike,
        skillbook: Skillbook,
        *,
        interval: int = 10,
    ) -> None:
        self.manager = manager
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx

        report = self.manager.get_similarity_report(self.skillbook)
        if report:
            logger.info(
                "DeduplicateStep: similarity report at sample %d:\n%s",
                ctx.global_sample_index,
                report,
            )
        return ctx
