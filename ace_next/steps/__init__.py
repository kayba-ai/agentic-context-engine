"""ACE pipeline steps — one class per file, plus the learning_tail helper."""

from __future__ import annotations

from pathlib import Path

from pipeline.protocol import StepProtocol

from ..protocols import (
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..core.skillbook import Skillbook

from .agent import AgentStep
from .apply import ApplyStep
from .checkpoint import CheckpointStep
from .deduplicate import DeduplicateStep
from .evaluate import EvaluateStep
from .observability import ObservabilityStep
from .opik import OPIK_AVAILABLE, OpikStep, register_opik_litellm_callback
from .persist import PersistStep
from .reflect import ReflectStep
from .tag import TagStep
from .update import UpdateStep

__all__ = [
    "AgentStep",
    "ApplyStep",
    "CheckpointStep",
    "DeduplicateStep",
    "EvaluateStep",
    "ObservabilityStep",
    "OPIK_AVAILABLE",
    "OpikStep",
    "PersistStep",
    "ReflectStep",
    "TagStep",
    "UpdateStep",
    "register_opik_litellm_callback",
    "learning_tail",
]


def learning_tail(
    reflector: ReflectorLike,
    skill_manager: SkillManagerLike,
    skillbook: Skillbook,
    *,
    dedup_manager: DeduplicationManagerLike | None = None,
    dedup_interval: int = 10,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 10,
) -> list[StepProtocol]:
    """Return the standard ACE learning steps.

    Use this when building custom integrations that provide their own
    execute step(s) but want the standard learning pipeline::

        steps = [
            MyCustomExecuteStep(my_agent),
            *learning_tail(reflector, skill_manager, skillbook),
        ]

    The returned list is always:
        [ReflectStep, TagStep, UpdateStep, ApplyStep]
    with optional DeduplicateStep and CheckpointStep appended when
    the corresponding config is provided.
    """
    steps: list[StepProtocol] = [
        ReflectStep(reflector),
        TagStep(skillbook),
        UpdateStep(skill_manager),
        ApplyStep(skillbook),
    ]
    if dedup_manager:
        steps.append(
            DeduplicateStep(dedup_manager, skillbook, interval=dedup_interval)
        )
    if checkpoint_dir:
        steps.append(
            CheckpointStep(checkpoint_dir, skillbook, interval=checkpoint_interval)
        )
    return steps
