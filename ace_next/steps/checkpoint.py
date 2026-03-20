"""CheckpointStep — periodically saves the skillbook to disk."""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.context import ACEStepContext
from ..core.path_safety import safe_resolve, safe_resolve_within
from ..core.skillbook import Skillbook

logger = logging.getLogger(__name__)


class CheckpointStep:
    """Save the skillbook to disk at a configurable interval.

    Optional tail step appended by factory methods when ``checkpoint_dir``
    is provided.

    Stateless — uses ``ctx.global_sample_index`` for interval logic.
    Saves both a numbered checkpoint and a ``latest.json`` that is
    always overwritten with the most recent state.

    The checkpoint directory is resolved at construction time and all
    generated file paths are validated to stay within that directory,
    preventing directory-traversal attacks via ``checkpoint_dir``.
    """

    requires: frozenset[str] = frozenset({"global_sample_index"})
    provides: frozenset[str] = frozenset()

    def __init__(
        self,
        directory: str | Path,
        skillbook: Skillbook,
        *,
        interval: int = 10,
    ) -> None:
        self.directory = safe_resolve(directory)
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx

        self.directory.mkdir(parents=True, exist_ok=True)

        numbered = safe_resolve_within(
            self.directory / f"checkpoint_{ctx.global_sample_index}.json",
            self.directory,
        )
        latest = safe_resolve_within(
            self.directory / "latest.json",
            self.directory,
        )

        self.skillbook.save_to_file(str(numbered))
        self.skillbook.save_to_file(str(latest))

        logger.info(
            "CheckpointStep: saved checkpoint at sample %d → %s",
            ctx.global_sample_index,
            numbered,
        )
        return ctx
