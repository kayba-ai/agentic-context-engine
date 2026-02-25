"""PersistStep — writes the skillbook to an external file."""

from __future__ import annotations

from pathlib import Path

from ..core.skillbook import Skillbook

from ..core.context import ACEStepContext


class PersistStep:
    """Write the current skillbook to a target file after each sample.

    Integration-specific side-effect step — used by ClaudeCodeACE to
    persist learned strategies into the project's CLAUDE.md.

    Unlike CheckpointStep (which saves full JSON at intervals), PersistStep
    runs on every sample and writes in whatever format the target expects.
    """

    requires = frozenset({"skillbook"})
    provides = frozenset()

    def __init__(self, target_path: str | Path, skillbook: Skillbook) -> None:
        self.target_path = Path(target_path)
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        self.skillbook.save_to_file(str(self.target_path))
        return ctx
