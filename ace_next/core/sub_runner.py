"""SubRunner — base class for steps that run an internal Pipeline in a loop.

Subclasses provide five template methods:

- ``_build_inner_pipeline()``  → the per-iteration step sequence
- ``_build_initial_context()`` → first iteration's context
- ``_is_done(ctx)``            → termination predicate
- ``_extract_result(ctx)``     → pull the final result from the context
- ``_accumulate(ctx)``         → build next iteration's context from current

The loop runs up to *max_iterations* times, calling the inner pipeline
once per iteration.  If termination is not reached, ``_on_timeout`` is
called to produce a fallback result.

``SubRunner`` satisfies :class:`~pipeline.protocol.StepProtocol` — it
can be placed directly in a Pipeline.
"""

from __future__ import annotations

import abc
from typing import Any

from pipeline.context import StepContext
from pipeline.pipeline import Pipeline


class SubRunner(abc.ABC):
    """Base for steps that run an internal Pipeline in a loop.

    Concrete subclasses must set ``requires`` and ``provides`` (as
    ``frozenset[str]``) and implement the five template methods plus
    ``__call__``.
    """

    requires: frozenset[str]
    provides: frozenset[str]

    def __init__(self, *, max_iterations: int = 20) -> None:
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Template methods (override in subclass)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_inner_pipeline(self, **kwargs: Any) -> Pipeline:
        """Return the Pipeline to execute once per iteration."""

    @abc.abstractmethod
    def _build_initial_context(self, **kwargs: Any) -> StepContext:
        """Return the StepContext for the first iteration."""

    @abc.abstractmethod
    def _is_done(self, ctx: StepContext) -> bool:
        """Return True when the loop should stop (success)."""

    @abc.abstractmethod
    def _extract_result(self, ctx: StepContext) -> Any:
        """Pull the final result from the terminal context."""

    @abc.abstractmethod
    def _accumulate(self, ctx: StepContext) -> StepContext:
        """Build the next iteration's context from the current one."""

    def _on_timeout(self, last_ctx: StepContext, iteration: int, **kwargs: Any) -> Any:
        """Called when *max_iterations* is reached without termination.

        Default raises ``RuntimeError``.  Override to return a fallback.
        ``**kwargs`` are forwarded from ``run_loop()`` so subclasses can
        access call-local state without stashing it on ``self``.
        """
        raise RuntimeError(
            f"SubRunner hit max_iterations ({self.max_iterations}) "
            f"without terminating"
        )

    # ------------------------------------------------------------------
    # Loop driver
    # ------------------------------------------------------------------

    def run_loop(self, **kwargs: Any) -> Any:
        """Execute the iterative loop.

        Builds the inner pipeline once, then loops:
        context → pipeline → check → accumulate → repeat.
        """
        pipe = self._build_inner_pipeline(**kwargs)
        ctx = self._build_initial_context(**kwargs)

        for i in range(self.max_iterations):
            ctx = pipe(ctx)
            if self._is_done(ctx):
                return self._extract_result(ctx)
            ctx = self._accumulate(ctx)

        return self._on_timeout(ctx, self.max_iterations, **kwargs)

    # ------------------------------------------------------------------
    # StepProtocol — subclasses typically override __call__ to map
    # between outer context and the run_loop result.
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def __call__(self, ctx: StepContext) -> StepContext:
        """StepProtocol entry point — run the loop and attach results."""
