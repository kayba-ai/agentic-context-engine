"""Structural protocol and result type for the pipeline engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from .context import StepContext

Ctx = TypeVar("Ctx", bound=StepContext)


@runtime_checkable
class PipelineHook(Protocol):
    """Observation-only hook fired around each foreground step.

    Hooks observe execution — they do **not** transform data.  Both methods
    return ``None``; context flow stays exclusively in the step chain via
    ``requires``/``provides``.

    Hooks must not block the event loop.  Heavy work (HTTP, disk) should be
    dispatched to a background task or queue.

    If a hook raises, the pipeline logs the error and continues.  A broken
    hook must never kill the pipeline.
    """

    def before_step(self, step_name: str, ctx: StepContext) -> None: ...
    def after_step(self, step_name: str, ctx: StepContext) -> None: ...


@runtime_checkable
class StepProtocol(Protocol[Ctx]):
    """Structural protocol that every step (and Pipeline/Branch) must satisfy.

    Generic over the context type — use ``StepProtocol[ACEStepContext]`` to
    type-check steps that accept a specific ``StepContext`` subclass.

    ``@runtime_checkable`` lets the pipeline validator use
    ``isinstance(step, StepProtocol)`` at construction time to give a clear
    error if a step is missing required attributes.
    """

    requires: frozenset[str]
    provides: frozenset[str]

    def __call__(self, ctx: Ctx) -> Ctx: ...


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
