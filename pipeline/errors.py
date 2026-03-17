"""Pipeline error types."""

from __future__ import annotations

import threading


class PipelineOrderError(Exception):
    """A step requires a field that no earlier step provides."""


class PipelineConfigError(Exception):
    """Invalid pipeline wiring.

    Examples:
    - More than one ``async_boundary = True`` step in the same pipeline.
    - An ``async_boundary = True`` step inside a Branch child.
    """


class BranchError(Exception):
    """One or more branch pipelines failed.

    All branches always run to completion before this is raised.
    ``failures`` contains the full list of exceptions — one per failed branch.
    """

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = failures
        super().__init__(
            f"{len(failures)} branch(es) failed: "
            + "; ".join(type(e).__name__ for e in failures)
        )


class PipelineCancelled(Exception):
    """A ``cancel_token`` was triggered between steps.

    Surfaces in ``SampleResult.error`` — never propagated to the caller of
    ``run()`` / ``run_async()``.  Callers check for this type to distinguish
    cancellation from step failures.
    """


class CancellationToken:
    """Thread-safe cancellation signal for pipeline runs.

    Create a fresh token per ``run()`` / ``run_async()`` invocation.  The
    pipeline checks ``is_cancelled`` between steps; call ``cancel()`` from
    any thread to stop processing.

    Uses ``threading.Event`` so it is safe to signal from a web endpoint
    handler, a background task, or a signal handler.
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation.  Thread-safe, idempotent."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()
