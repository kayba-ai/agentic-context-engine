"""ApplyStep — applies update operations to the skillbook."""

from __future__ import annotations

from ..skillbook import Skillbook

from ..context import ACEStepContext


class ApplyStep:
    """Apply the update batch to the real Skillbook.

    Side-effect step — mutates ``self.skillbook`` (injected via constructor).
    Separated from UpdateStep so that UpdateStep can be tested without
    mutating a skillbook, and ApplyStep can be tested with a mock batch.
    """

    requires = frozenset({"skill_manager_output"})
    provides = frozenset()

    max_workers = 1

    def __init__(self, skillbook: Skillbook) -> None:
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        self.skillbook.apply_update(ctx.skill_manager_output)
        return ctx
