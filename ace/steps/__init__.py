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
from .apply import ApplyStep
from .attach_insight_sources import AttachInsightSourcesStep
from .checkpoint import CheckpointStep
from .deduplicate import DeduplicateStep
from .evaluate import EvaluateStep
from .export_markdown import ExportSkillbookMarkdownStep
from .load_traces import LoadTracesStep
from .observability import ObservabilityStep
from .persist import PersistStep
from .reflect import ReflectStep
from .update import UpdateStep

__all__ = [
    "AgentStep",
    "ApplyStep",
    "AttachInsightSourcesStep",
    "CheckpointStep",
    "DeduplicateStep",
    "EvaluateStep",
    "ExportSkillbookMarkdownStep",
    "LoadTracesStep",
    "ObservabilityStep",
    "PersistStep",
    "ReflectStep",
    "UpdateStep",
    "learning_tail",
]


def _reflect_step(reflector: ReflectorLike) -> StepProtocol[ACEStepContext]:
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
    ``provides = {'reflections'}``, followed by ``UpdateStep``,
    ``AttachInsightSourcesStep``, and ``ApplyStep``. Optional
    ``DeduplicateStep`` and ``CheckpointStep`` are appended when configured.
    """
    steps: list[StepProtocol[ACEStepContext]] = [
        _reflect_step(reflector),
        UpdateStep(skill_manager),
        AttachInsightSourcesStep(),
        ApplyStep(skillbook),
    ]
    if dedup_manager:
        steps.append(DeduplicateStep(dedup_manager, skillbook, interval=dedup_interval))
    if checkpoint_dir:
        steps.append(
            CheckpointStep(checkpoint_dir, skillbook, interval=checkpoint_interval)
        )
    return steps
