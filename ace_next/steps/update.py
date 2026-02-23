"""UpdateStep — generates skillbook update operations from a reflection."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..context import ACEStepContext
from ..outputs import ReflectorOutput, SkillManagerOutput


@runtime_checkable
class SkillManagerLike(Protocol):
    """Protocol for SkillManager-like objects.

    Satisfied by ``ace.roles.SkillManager``.
    """

    def update_skills(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput: ...


class UpdateStep:
    """Run the SkillManager role to produce update operations.

    Pure — generates an ``UpdateBatch`` from the reflection and the current
    skillbook state.  Does not mutate the skillbook.  ``max_workers = 1``
    because the skill manager reads the current skillbook state and
    concurrent calls would see stale data.
    """

    requires = frozenset({"reflection", "skillbook"})
    provides = frozenset({"skill_manager_output"})

    max_workers = 1

    def __init__(self, skill_manager: SkillManagerLike) -> None:
        self.skill_manager = skill_manager

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        # Build progress string from context counters
        progress = f"Epoch {ctx.epoch}/{ctx.total_epochs}"
        if ctx.total_steps is not None:
            progress += f", sample {ctx.step_index}/{ctx.total_steps}"

        # Build question_context from the trace when available
        question_context = ""
        if isinstance(ctx.trace, dict):
            q = ctx.trace.get("question", "")
            c = ctx.trace.get("context", "")
            question_context = f"{q}\n{c}".strip() if c else q

        output = self.skill_manager.update_skills(
            reflection=ctx.reflection,
            skillbook=ctx.skillbook,
            question_context=question_context,
            progress=progress,
        )
        return ctx.replace(skill_manager_output=output.update)
