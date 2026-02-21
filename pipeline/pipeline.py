"""Pipeline — concrete, composable, sequential step runner."""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .branch import Branch, MergeStrategy
from .context import StepContext
from .errors import PipelineConfigError, PipelineOrderError
from .protocol import SampleResult


# ---------------------------------------------------------------------------
# Per-step-class background executor registry
# ---------------------------------------------------------------------------

_executor_lock = threading.Lock()


def _get_class_executor(step_cls: type) -> ThreadPoolExecutor:
    """Return the class-level ThreadPoolExecutor for *step_cls*, creating it lazily.

    The executor is stored on the class itself (``step_cls._executor``) so it
    is shared across all pipeline instances.  ``max_workers`` defaults to 1
    if not declared on the class.
    """
    if not hasattr(step_cls, "_executor") or step_cls._executor is None:
        with _executor_lock:
            # Double-checked locking
            if not hasattr(step_cls, "_executor") or step_cls._executor is None:
                max_workers = getattr(step_cls, "max_workers", 1)
                step_cls._executor = ThreadPoolExecutor(max_workers=max_workers)
    return step_cls._executor


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Ordered sequence of steps.  Satisfies StepProtocol — can be nested.

    Build via the fluent API::

        pipe = (
            Pipeline()
            .then(AgentStep())
            .then(EvaluateStep())
            .then(ReflectStep())   # ReflectStep.async_boundary = True
            .then(UpdateStep())
        )

    Fan-out across samples::

        results = pipe.run(samples, workers=4)

    ``requires`` and ``provides`` are inferred from the step chain and kept
    up-to-date as steps are added, so a ``Pipeline`` can itself be used as a
    step inside another pipeline without extra annotation.
    """

    def __init__(self, steps: list | None = None) -> None:
        self._steps: list = list(steps or [])
        self.requires, self.provides = self._infer_contracts(self._steps)
        self._validate_steps(self._steps)

        # Background thread tracking (per Pipeline instance)
        self._bg_threads: list[threading.Thread] = []
        self._bg_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Contract inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_contracts(steps: list) -> tuple[frozenset, frozenset]:
        """Compute (requires, provides) for the full step chain.

        ``requires`` — fields the pipeline needs from the outside
                       (what its first steps need that no earlier inner
                       step provides).
        ``provides`` — union of everything any inner step writes.
        """
        provided_so_far: set[str] = set()
        external_requires: set[str] = set()
        for step in steps:
            step_requires = set(getattr(step, "requires", frozenset()))
            step_provides = set(getattr(step, "provides", frozenset()))
            external_requires |= step_requires - provided_so_far
            provided_so_far |= step_provides
        return frozenset(external_requires), frozenset(provided_so_far)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_steps(steps: list) -> None:
        """Raise PipelineOrderError or PipelineConfigError for invalid wiring.

        Order check:
            If step B requires field X, and field X is produced by some step
            in the pipeline but that step appears *after* B, raise
            ``PipelineOrderError``.  Fields not produced by any step in the
            pipeline are treated as external inputs — no error.

        Config checks:
            - More than one ``async_boundary = True`` step in the same pipeline.
            - Any ``async_boundary = True`` step inside a Branch child.
            - Warning (not error) when ``async_boundary`` is set on a nested
              Pipeline (the boundary is ignored when the pipeline runs as a step).
        """
        # Pre-compute all fields ever produced internally
        all_provided_internally: set[str] = set()
        for step in steps:
            all_provided_internally |= set(getattr(step, "provides", frozenset()))

        provided_so_far: set[str] = set()
        boundary_count = 0

        for step in steps:
            step_requires = set(getattr(step, "requires", frozenset()))
            step_provides = set(getattr(step, "provides", frozenset()))

            # Ordering: field is produced internally but not yet available
            out_of_order = (step_requires & all_provided_internally) - provided_so_far
            if out_of_order:
                raise PipelineOrderError(
                    f"{type(step).__name__} requires {out_of_order!r} but these "
                    f"are produced by a later step — check step ordering."
                )

            provided_so_far |= step_provides

            # async_boundary: only one per pipeline
            if getattr(step, "async_boundary", False):
                boundary_count += 1
                if boundary_count > 1:
                    raise PipelineConfigError(
                        f"Only one async_boundary step is allowed per pipeline; "
                        f"{type(step).__name__} is a duplicate."
                    )

            # async_boundary inside a Branch child is forbidden
            if isinstance(step, Branch):
                for child in step.pipelines:
                    for child_step in getattr(child, "_steps", []):
                        if getattr(child_step, "async_boundary", False):
                            raise PipelineConfigError(
                                f"async_boundary is not allowed inside a Branch "
                                f"child (found on {type(child_step).__name__})."
                            )

            # Warn when async_boundary is set on a nested Pipeline (ignored)
            if isinstance(step, Pipeline) and getattr(step, "async_boundary", False):
                warnings.warn(
                    f"async_boundary declared on a nested Pipeline "
                    f"({type(step).__name__}) is ignored — the boundary only "
                    f"fires when the pipeline is used as a top-level runner.",
                    stacklevel=4,
                )

    # ------------------------------------------------------------------
    # Fluent builder
    # ------------------------------------------------------------------

    def then(self, step: object) -> "Pipeline":
        """Append *step* and return ``self`` for chaining."""
        new_steps = self._steps + [step]
        # Validate before mutating so errors are raised immediately
        self._validate_steps(new_steps)
        self._steps = new_steps
        self.requires, self.provides = self._infer_contracts(self._steps)
        return self

    def branch(
        self,
        *pipelines: object,
        merge: MergeStrategy | Any = MergeStrategy.RAISE_ON_CONFLICT,
    ) -> "Pipeline":
        """Append a Branch step and return ``self`` for chaining."""
        return self.then(Branch(*pipelines, merge=merge))

    # ------------------------------------------------------------------
    # __call__ — for use as a nested step
    # ------------------------------------------------------------------

    def __call__(self, ctx: StepContext) -> StepContext:
        """Run all steps sequentially (sync).

        When used as a nested step inside another pipeline, ``async_boundary``
        markers are **ignored** (a warning is already emitted at construction
        time).  All steps — sync and async — are executed to completion before
        returning.
        """
        for step in self._steps:
            if asyncio.iscoroutinefunction(step.__call__):
                # Run the coroutine in a new event loop (safe in non-async contexts)
                ctx = asyncio.run(step(ctx))
            elif isinstance(step, Branch):
                # Branch.__call__ is sync (ThreadPoolExecutor)
                ctx = step(ctx)
            else:
                ctx = step(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Async_boundary helpers
    # ------------------------------------------------------------------

    def _find_boundary_index(self) -> int | None:
        """Return the index of the first async_boundary step, or None."""
        for i, step in enumerate(self._steps):
            if getattr(step, "async_boundary", False):
                return i
        return None

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------

    def _submit_background(
        self,
        ctx: StepContext,
        background_steps: list,
        result: SampleResult,
    ) -> None:
        """Run *background_steps* sequentially in a background thread.

        Each step is submitted to its own class-level executor so concurrency
        across samples is controlled by ``max_workers`` on the step class —
        independent of how many pipeline instances or background tails are
        running.

        ``result`` is mutated in-place when the tail completes (or fails).
        """

        def run_tail() -> None:
            current_ctx = ctx
            for step in background_steps:
                step_cls = type(step)
                executor = _get_class_executor(step_cls)
                try:
                    # Submit to per-step-class pool; block until slot is free
                    future = executor.submit(step, current_ctx)
                    current_ctx = future.result()
                except Exception as exc:
                    result.error = exc
                    result.failed_at = step_cls.__name__
                    result.output = None
                    return
            result.output = current_ctx

        t = threading.Thread(target=run_tail, daemon=True, name="pipeline-bg")
        with self._bg_lock:
            self._bg_threads.append(t)
        t.start()

    def wait_for_background(self, timeout: float | None = None) -> None:
        """Block until all background tasks submitted by this pipeline finish.

        Raises ``TimeoutError`` if *timeout* seconds elapse before all tasks
        complete.  Completed threads are removed from the tracking list.
        """
        with self._bg_lock:
            threads = list(self._bg_threads)

        deadline = None if timeout is None else time.monotonic() + timeout

        for t in threads:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Background pipeline steps did not drain within timeout."
                    )
                t.join(timeout=remaining)
                if t.is_alive():
                    raise TimeoutError(
                        "Background pipeline steps did not drain within timeout."
                    )
            else:
                t.join()

        # Remove completed threads
        with self._bg_lock:
            self._bg_threads = [t for t in self._bg_threads if t.is_alive()]

    # ------------------------------------------------------------------
    # run() — sync entry point
    # ------------------------------------------------------------------

    def run(
        self, samples: Any, workers: int = 1
    ) -> list[SampleResult]:
        """Process *samples* through the pipeline (sync entry point).

        Splits at the first ``async_boundary`` step:
        - Foreground steps run in the calling context (with up to ``workers``
          samples in parallel via a semaphore inside the event loop).
        - Background steps are submitted to per-step-class executors and run
          asynchronously.  Call ``wait_for_background()`` to block until they
          finish and ``SampleResult`` objects are fully populated.

        Every sample produces exactly one ``SampleResult`` — nothing is
        dropped silently.
        """
        return asyncio.run(self.run_async(samples, workers=workers))

    # ------------------------------------------------------------------
    # run_async() — async entry point
    # ------------------------------------------------------------------

    async def run_async(
        self, samples: Any, workers: int = 1
    ) -> list[SampleResult]:
        """Async entry point; use ``await pipe.run_async(samples)`` from
        coroutine contexts (e.g. inside browser-use tasks)."""
        boundary_idx = self._find_boundary_index()
        if boundary_idx is None:
            foreground_steps = self._steps
            background_steps: list = []
        else:
            foreground_steps = self._steps[:boundary_idx]
            background_steps = self._steps[boundary_idx:]

        sem = asyncio.Semaphore(workers)

        async def process_one(sample: Any) -> SampleResult:
            async with sem:
                ctx = StepContext(sample=sample)
                result = SampleResult(
                    sample=sample, output=None, error=None, failed_at=None
                )
                last_step_name: str | None = None
                try:
                    for step in foreground_steps:
                        last_step_name = type(step).__name__
                        if asyncio.iscoroutinefunction(step.__call__):
                            ctx = await step(ctx)
                        elif hasattr(step, "__call_async__"):
                            ctx = await step.__call_async__(ctx)
                        else:
                            ctx = await asyncio.to_thread(step, ctx)
                except Exception as exc:
                    result.error = exc
                    result.failed_at = last_step_name
                    return result

                if background_steps:
                    # Fire and forget — result updated by background thread
                    self._submit_background(ctx, background_steps, result)
                else:
                    result.output = ctx

                return result

        return list(await asyncio.gather(*[process_one(s) for s in samples]))
