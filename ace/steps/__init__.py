"""ACE pipeline steps — one class per file, plus the learning_tail helper."""

from __future__ import annotations

from pathlib import Path

from pipeline.protocol import StepProtocol

from ..core.context import ACEStepContext
from ..protocols import (
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..core.skillbook import Skillbook

from .agent import AgentStep
from .checkpoint import CheckpointStep
from .deduplicate import DeduplicateStep
from .evaluate import EvaluateStep
from .export_markdown import ExportSkillbookMarkdownStep
from .load_traces import LoadTracesStep
from .observability import ObservabilityStep
from .persist import PersistStep
from .reflect import ReflectStep
from .reflection_ensemble import ReflectionEnsembleStep
from .update import UpdateStep

__all__ = [
    "AgentStep",
    "CheckpointStep",
    "DeduplicateStep",
    "EvaluateStep",
    "ExportSkillbookMarkdownStep",
    "LoadTracesStep",
    "ObservabilityStep",
    "PersistStep",
    "ReflectStep",
    "ReflectionEnsembleStep",
    "UpdateStep",
    "learning_tail",
]


def _reflect_step(
    reflector: ReflectorLike,
    *,
    reflection_ensemble_size: int = 1,
    reflection_ensemble_workers: int | None = None,
) -> StepProtocol[ACEStepContext]:
    if reflection_ensemble_size != 1:
        return ReflectionEnsembleStep(
            reflector,
            ensemble_size=reflection_ensemble_size,
            workers=reflection_ensemble_workers,
        )

    provides = getattr(reflector, "provides", ())
    if callable(reflector) and "reflections" in provides:
        return reflector  # type: ignore[return-value]
    return ReflectStep(reflector)


def learning_tail(
    reflector: ReflectorLike,
    skill_manager: SkillManagerLike,
    skillbook: Skillbook,
    *,
    dedup_manager: DeduplicationManagerLike | None = None,
    dedup_interval: int = 10,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 10,
    reflection_ensemble_size: int = 1,
    reflection_ensemble_workers: int | None = None,
) -> list[StepProtocol[ACEStepContext]]:
    """Return the standard ACE learning steps.

    Use this when building custom integrations that provide their own
    execute step(s) but want the standard learning pipeline::

        steps = [
            MyCustomExecuteStep(my_agent),
            *learning_tail(reflector, skill_manager, skillbook),
        ]

    The returned list starts with either ``ReflectStep`` or the provided
    reflector itself when it already satisfies the step protocol and exposes
    ``provides = {'reflections'}``, followed by ``UpdateStep``. The agentic
    SkillManager mutates the skillbook directly through its tools, so no
    ``ApplyStep`` follows. Set ``reflection_ensemble_size > 1`` to run the
    same reflector multiple times on each trace and pass all resulting
    reflections to one ``UpdateStep``. Optional ``DeduplicateStep`` and
    ``CheckpointStep`` are appended when configured.
    """
    steps: list[StepProtocol[ACEStepContext]] = [
        _reflect_step(
            reflector,
            reflection_ensemble_size=reflection_ensemble_size,
            reflection_ensemble_workers=reflection_ensemble_workers,
        ),
        UpdateStep(skill_manager, skillbook),
    ]
    if dedup_manager:
        steps.append(DeduplicateStep(dedup_manager, skillbook, interval=dedup_interval))
    if checkpoint_dir:
        steps.append(
            CheckpointStep(checkpoint_dir, skillbook, interval=checkpoint_interval)
        )
    return steps
