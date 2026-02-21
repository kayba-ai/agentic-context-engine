"""Immutable step context — the single object that flows through every step."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class StepContext:
    """Frozen context object passed from step to step.

    Named fields cover every concept shared across ACE pipelines.
    Integration-specific data goes in ``metadata`` so named fields never grow
    as integrations are added.

    Steps never mutate the incoming context — they call ``.replace()`` to
    produce a new one.
    """

    # Core input
    sample: Any = None
    skillbook: Any = None
    environment: Any = None

    # Epoch / progress counters (set by the runner, not by steps)
    epoch: int = 1
    total_epochs: int = 1
    step_index: int = 0
    total_steps: int = 0

    # Rolling window of past reflections (tuple for immutability)
    recent_reflections: tuple = field(default_factory=tuple)

    # Named outputs produced by the four ACE steps
    agent_output: Any = None
    environment_result: Any = None
    reflection: Any = None
    skill_manager_output: Any = None

    # Integration-specific payload — always goes here, never as a new named field
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Coerce plain dict → MappingProxyType so mutation is a hard runtime error
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(self.metadata))
        # Coerce list/other iterables → tuple for immutability
        if not isinstance(self.recent_reflections, tuple):
            object.__setattr__(
                self, "recent_reflections", tuple(self.recent_reflections)
            )

    def replace(self, **changes: Any) -> "StepContext":
        """Return a new StepContext with the given fields replaced."""
        return dataclasses.replace(self, **changes)
