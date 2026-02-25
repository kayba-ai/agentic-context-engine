"""DeduplicateStep — periodically consolidates similar skills."""

from __future__ import annotations

import logging

from ..core.context import ACEStepContext
from ..protocols import DeduplicationManagerLike
from ..core.skillbook import Skillbook

logger = logging.getLogger(__name__)


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
