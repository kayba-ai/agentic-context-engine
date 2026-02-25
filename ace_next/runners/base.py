"""ACERunner — shared runner infrastructure for all ACE runners."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.skillbook import Skillbook

logger = logging.getLogger(__name__)


class ACERunner:
    """Shared runner infrastructure for all ACE runners.

    Composes a ``Pipeline`` (does not extend it).  Manages the epoch loop
    and delegates per-sample iteration, error isolation, foreground/background
    split, and concurrent workers to ``Pipeline.run()``.

    Subclasses override two methods:

    - ``run()`` — public API with a subclass-specific signature.
    - ``_build_context()`` — maps a single input item to ``ACEStepContext``.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        skillbook: Skillbook,
    ) -> None:
        self.pipeline = pipeline
        self.skillbook = skillbook

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the current skillbook to disk."""
        self.skillbook.save_to_file(path)

    def wait_for_background(self, timeout: float | None = None) -> None:
        """Block until all background learning tasks complete.

        Delegates to ``Pipeline.wait_for_background()``.  Call after
        ``run(wait=False)`` before saving the skillbook or reading final
        results.
        """
        self.pipeline.wait_for_background(timeout)

    @property
    def learning_stats(self) -> dict[str, int]:
        """Return background learning progress.

        Delegates to ``Pipeline.background_stats()``.
        """
        return self.pipeline.background_stats()

    # ------------------------------------------------------------------
    # Generic epoch loop (called by subclasses)
    # ------------------------------------------------------------------

    def _run(
        self,
        items: Sequence[Any] | Iterable[Any],
        *,
        epochs: int,
        wait: bool = True,
        **kwargs: Any,
    ) -> list[SampleResult]:
        """Generic run loop handling epochs and Iterable validation.

        Returns when ``wait=True`` (default).  Returns after foreground
        steps when ``wait=False`` — background learning continues.

        Raises ``ValueError`` if ``epochs > 1`` and *items* is not a
        ``Sequence``.
        """
        if epochs > 1 and not isinstance(items, Sequence):
            raise ValueError(
                "Multi-epoch requires a Sequence, not a consumed Iterable."
            )

        results: list[SampleResult] = []
        n: int | None = len(items) if isinstance(items, Sequence) else None

        for epoch in range(1, epochs + 1):
            contexts: list[ACEStepContext] = [
                self._build_context(
                    item,
                    epoch=epoch,
                    total_epochs=epochs,
                    index=idx,
                    total=n,
                    global_sample_index=(
                        (epoch - 1) * n + idx if n is not None else idx
                    ),
                    **kwargs,
                )
                for idx, item in enumerate(items, start=1)
            ]
            epoch_results = self.pipeline.run(contexts)
            results.extend(epoch_results)

        if wait:
            self.pipeline.wait_for_background()

        return results

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _build_context(
        self,
        item: Any,
        *,
        epoch: int,
        total_epochs: int,
        index: int,
        total: int | None,
        global_sample_index: int,
        **kwargs: Any,
    ) -> ACEStepContext:
        """Map a single input item to an ``ACEStepContext``.

        Must be overridden by subclasses.  Stateless — depends only on
        the item and the provided counters.
        """
        raise NotImplementedError
