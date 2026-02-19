"""Structural protocol and result type for the pipeline engine."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .context import StepContext


@runtime_checkable
class StepProtocol(Protocol):
    """Structural protocol that every step (and Pipeline/Branch) must satisfy.

    ``AbstractSet[str]`` accepts both ``set`` and ``frozenset`` — steps may
    declare plain set literals; the pipeline normalises them to ``frozenset``
    at construction time before contract validation.

    ``@runtime_checkable`` lets the pipeline validator use
    ``isinstance(step, StepProtocol)`` at construction time to give a clear
    error if a step is missing required attributes.
    """

    requires: AbstractSet[str]
    provides: AbstractSet[str]

    def __call__(self, ctx: StepContext) -> StepContext: ...


@dataclass
class SampleResult:
    """Outcome for one sample after the pipeline has run.

    Every sample produces exactly one ``SampleResult`` — nothing is dropped
    silently.  After ``run()`` returns, inspect ``error`` / ``failed_at`` to
    detect failures; ``output`` is ``None`` whenever a step raised.

    For background steps (after ``async_boundary``), ``output`` / ``error``
    may still be ``None`` when ``run()`` returns.  Call
    ``pipeline.wait_for_background()`` to block until all background work
    completes and results are finalised.

    When a ``Branch`` step fails, ``failed_at == "Branch"`` and ``cause``
    holds the inner exception from the failing branch.
    """

    sample: Any
    output: StepContext | None
    error: Exception | None
    failed_at: str | None
    cause: Exception | None = None
