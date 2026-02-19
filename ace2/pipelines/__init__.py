"""ACE pipeline runners and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline import Pipeline

from ace2.steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep

from .offline import OfflineACE
from .online import OnlineACE

if TYPE_CHECKING:
    from ace.roles import Agent, Reflector, SkillManager


def ace_pipeline(
    agent: "Agent",
    reflector: "Reflector",
    skill_manager: "SkillManager",
    *,
    max_refinement_rounds: int = 1,
    reflection_window: int = 3,
) -> Pipeline:
    """Wire the standard four-step ACE pipeline.

    ::

        AgentStep → EvaluateStep → ReflectStep → UpdateStep
    """
    return (
        Pipeline()
        .then(AgentStep(agent))
        .then(EvaluateStep())
        .then(
            ReflectStep(
                reflector,
                max_refinement_rounds=max_refinement_rounds,
                reflection_window=reflection_window,
            )
        )
        .then(UpdateStep(skill_manager))
    )


__all__ = ["ace_pipeline", "OfflineACE", "OnlineACE"]
