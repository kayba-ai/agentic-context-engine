"""Concrete pipeline steps for the ACE framework.

Each step is a callable class that receives a :class:`~ace.pipeline.base.StepContext`,
performs one focused unit of work, and returns the (mutated) context.

Default step chain::

    AgentStep → EvaluateStep → ReflectStep → UpdateStep

Steps can be replaced or extended by passing a custom ``steps`` list to any
:class:`~ace.pipeline.base.ACEPipeline` subclass.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..roles import Agent, Reflector, SkillManager

from .base import StepContext

logger = logging.getLogger(__name__)


class AgentStep:
    """Runs the Agent to produce an answer from the current skillbook.

    **Reads:** ``sample``, ``skillbook``, ``recent_reflections``

    **Writes:** ``ctx.agent_output``
    """

    def __init__(self, agent: "Agent") -> None:
        self.agent = agent

    def __call__(self, ctx: StepContext) -> StepContext:
        ctx.agent_output = self.agent.generate(
            question=ctx.sample.question,
            context=ctx.sample.context,
            skillbook=ctx.skillbook,
            reflection="\n---\n".join(ctx.recent_reflections),
            sample=ctx.sample,
        )
        return ctx


class EvaluateStep:
    """Evaluates the agent's output against the task environment.

    **Reads:** ``sample``, ``agent_output``, ``environment``

    **Writes:** ``ctx.environment_result``
    """

    def __call__(self, ctx: StepContext) -> StepContext:
        ctx.environment_result = ctx.environment.evaluate(
            ctx.sample, ctx.agent_output
        )
        return ctx


class ReflectStep:
    """Runs the Reflector to analyse what worked and what didn't.

    Also applies skill tags from the reflection to the skillbook and
    updates the rolling ``recent_reflections`` window on the context.

    **Reads:** ``sample``, ``agent_output``, ``environment_result``,
    ``skillbook``, ``recent_reflections``

    **Writes:** ``ctx.reflection``, ``ctx.recent_reflections``,
    ``ctx.skillbook`` (tags only)
    """

    def __init__(
        self,
        reflector: "Reflector",
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
    ) -> None:
        self.reflector = reflector
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window

    def __call__(self, ctx: StepContext) -> StepContext:
        ctx.reflection = self.reflector.reflect(
            question=ctx.sample.question,
            agent_output=ctx.agent_output,
            skillbook=ctx.skillbook,
            ground_truth=ctx.environment_result.ground_truth,
            feedback=ctx.environment_result.feedback,
            max_refinement_rounds=self.max_refinement_rounds,
        )

        # Apply skill tags to the shared skillbook
        for tag in ctx.reflection.skill_tags:
            try:
                ctx.skillbook.tag_skill(tag.id, tag.tag)
            except ValueError:
                continue

        # Update rolling recent-reflections window
        serialized = json.dumps(ctx.reflection.raw, ensure_ascii=False)
        ctx.recent_reflections = [
            *ctx.recent_reflections,
            serialized,
        ][-self.reflection_window :]

        return ctx


class UpdateStep:
    """Runs the SkillManager to update the skillbook with new strategies.

    **Reads:** ``reflection``, ``skillbook``, ``sample``,
    ``environment_result``, ``epoch``/``step`` counters

    **Writes:** ``ctx.skill_manager_output``, ``ctx.skillbook``
    (via ``apply_update``)
    """

    def __init__(self, skill_manager: "SkillManager") -> None:
        self.skill_manager = skill_manager

    def __call__(self, ctx: StepContext) -> StepContext:
        question_context = "\n".join(
            [
                f"question: {ctx.sample.question}",
                f"context: {ctx.sample.context}",
                f"metadata: {json.dumps(ctx.sample.metadata)}",
                f"feedback: {ctx.environment_result.feedback}",
                f"ground_truth: {ctx.environment_result.ground_truth}",
            ]
        )
        progress = (
            f"epoch {ctx.epoch}/{ctx.total_epochs} · "
            f"sample {ctx.step_index}/{ctx.total_steps}"
        )

        ctx.skill_manager_output = self.skill_manager.update_skills(
            reflection=ctx.reflection,
            skillbook=ctx.skillbook,
            question_context=question_context,
            progress=progress,
        )
        ctx.skillbook.apply_update(ctx.skill_manager_output.update)
        return ctx
