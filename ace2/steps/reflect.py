"""ReflectStep — analyses what went right/wrong and tags skills."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pipeline import StepContext

if TYPE_CHECKING:
    from ace.roles import Reflector

logger = logging.getLogger(__name__)


class ReflectStep:
    """Pipeline step that wraps the ACE Reflector role.

    Calls ``reflector.reflect()`` with the agent output and environment
    feedback, applies skill tags to the skillbook, and appends the new
    reflection to the rolling window.

    This step declares ``async_boundary = True`` so that it (and every
    subsequent step) runs in a background thread pool, allowing the
    foreground Agent + Evaluate to return quickly.
    """

    requires = frozenset({"sample", "agent_output", "environment_result", "skillbook"})
    provides = frozenset({"reflection", "recent_reflections"})

    # --- async boundary: everything from here onward runs in the background ---
    async_boundary = True
    max_workers = 3

    def __init__(
        self,
        reflector: "Reflector",
        *,
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
    ) -> None:
        self.reflector = reflector
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window

    def __call__(self, ctx: StepContext) -> StepContext:
        reflection = self.reflector.reflect(
            question=ctx.sample.question,
            agent_output=ctx.agent_output,
            skillbook=ctx.skillbook,
            ground_truth=ctx.environment_result.ground_truth,
            feedback=ctx.environment_result.feedback,
            max_refinement_rounds=self.max_refinement_rounds,
        )

        # Apply skill tags to the (shared, mutable) skillbook
        for tag in reflection.skill_tags:
            try:
                ctx.skillbook.tag_skill(tag.id, tag.tag)
            except ValueError:
                continue

        # Append to rolling reflection window (immutable tuple in context)
        serialized = json.dumps(reflection.raw, ensure_ascii=False)
        window = ctx.recent_reflections + (serialized,)
        if len(window) > self.reflection_window:
            window = window[-self.reflection_window :]

        return ctx.replace(reflection=reflection, recent_reflections=window)
