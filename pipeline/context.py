"""Immutable step context — the single object that flows through every step."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Self


@dataclass(frozen=True)
class StepContext:
    """Frozen context object passed from step to step.

    The pipeline engine only requires ``sample`` and ``metadata``.  All
    domain-specific fields are added by subclassing — the engine never reads
    anything beyond these two fields.

    Consuming applications subclass ``StepContext`` to add named fields for
    concepts shared across their pipelines.  Integration-specific data goes
    in ``metadata`` to prevent field accumulation on the subclass.

    Steps never mutate the incoming context — they call ``.replace()`` to
    produce a new one.
    """

    sample: Any = None
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Coerce plain dict → MappingProxyType so mutation is a hard runtime error
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(self.metadata))

    def replace(self, **changes: Any) -> Self:
        """Return a new context with the given fields replaced."""
        return dataclasses.replace(self, **changes)
