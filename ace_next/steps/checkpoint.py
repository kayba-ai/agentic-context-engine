"""CheckpointStep — periodically saves the skillbook to disk."""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.skillbook import Skillbook

from ..core.context import ACEStepContext

logger = logging.getLogger(__name__)


class CheckpointStep:
    """Save the skillbook to disk at a configurable interval.

    Optional tail step appended by factory methods when ``checkpoint_dir``
    is provided.

    Stateless — uses ``ctx.global_sample_index`` for interval logic.
    Saves both a numbered checkpoint and a ``latest.json`` that is
    always overwritten with the most recent state.
    """

    requires = frozenset({"global_sample_index"})
    provides = frozenset()

    def __init__(
        self,
        directory: str | Path,
        skillbook: Skillbook,
        *,
        interval: int = 10,
    ) -> None:
        self.directory = Path(directory)
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx

        self.directory.mkdir(parents=True, exist_ok=True)

        numbered = self.directory / f"checkpoint_{ctx.global_sample_index}.json"
        latest = self.directory / "latest.json"

        self.skillbook.save_to_file(str(numbered))
        self.skillbook.save_to_file(str(latest))

        logger.info(
            "CheckpointStep: saved checkpoint at sample %d → %s",
            ctx.global_sample_index,
            numbered,
        )
        return ctx
