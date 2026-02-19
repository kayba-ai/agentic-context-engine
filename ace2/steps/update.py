"""UpdateStep — applies skillbook mutations from the SkillManager."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pipeline import StepContext

if TYPE_CHECKING:
    from ace.roles import SkillManager

logger = logging.getLogger(__name__)


class UpdateStep:
    """Pipeline step that wraps the ACE SkillManager role.

    Calls ``skill_manager.update_skills()`` with the reflection output,
    attaches insight-source provenance metadata to each operation, then
    applies the resulting update batch to the skillbook in-place.

    ``max_workers = 1`` ensures only one SkillManager mutates the skillbook
    at a time (serialised writes), even when multiple ReflectSteps run in
    parallel upstream.
    """

    requires = frozenset({"reflection", "skillbook", "sample", "environment_result", "agent_output"})
    provides = frozenset({"skill_manager_output"})

    max_workers = 1

    def __init__(self, skill_manager: "SkillManager") -> None:
        self.skill_manager = skill_manager

    def __call__(self, ctx: StepContext) -> StepContext:
        # Build question context string (mirrors ACEBase._question_context)
        question_context = "\n".join([
            f"question: {ctx.sample.question}",
            f"context: {ctx.sample.context}",
            f"metadata: {json.dumps(ctx.sample.metadata)}",
            f"feedback: {ctx.environment_result.feedback}",
            f"ground_truth: {ctx.environment_result.ground_truth}",
        ])

        # Build progress string from context counters
        progress = (
            f"epoch {ctx.epoch}/{ctx.total_epochs} "
            f"· sample {ctx.step_index}/{ctx.total_steps}"
        )

        skill_manager_output = self.skill_manager.update_skills(
            reflection=ctx.reflection,
            skillbook=ctx.skillbook,
            question_context=question_context,
            progress=progress,
        )

        # Attach insight-source provenance before applying
        from ace.insight_source import build_insight_source

        build_insight_source(
            sample_question=ctx.sample.question,
            epoch=ctx.epoch,
            step=ctx.step_index,
            error_identification=ctx.reflection.error_identification,
            agent_output=ctx.agent_output,
            reflection=ctx.reflection,
            operations=skill_manager_output.update.operations,
            sample_id=getattr(ctx.sample, "id", None),
        )

        # Apply updates to the shared skillbook (in-place mutation)
        ctx.skillbook.apply_update(skill_manager_output.update)

        return ctx.replace(skill_manager_output=skill_manager_output)
