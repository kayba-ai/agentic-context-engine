"""TraceAnalyser — batch learning from pre-recorded execution traces."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..context import ACEStepContext, SkillbookView
from ..protocols import (
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..skillbook import Skillbook
from ..steps import learning_tail
from .base import ACERunner


class TraceAnalyser(ACERunner):
    """Analyse pre-recorded traces to build a skillbook.

    Runs the learning tail only — Reflect, Tag, Update, Apply — with
    optional deduplication and checkpoint steps.  No AgentStep, no
    EvaluateStep.

    Accepts raw trace objects of any type.  They are placed directly on
    ``ctx.trace`` for the Reflector to interpret.

    Use when you have execution logs from an external system (browser-use
    ``AgentHistoryList``, LangChain intermediate steps, Claude Code
    transcripts) and want to build or refine a skillbook from historical
    data.
    """

    @classmethod
    def from_roles(
        cls,
        *,
        reflector: ReflectorLike,
        skill_manager: SkillManagerLike,
        skillbook: Skillbook | None = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
    ) -> TraceAnalyser:
        """Construct from pre-built role instances.

        Args:
            reflector: Reflector role for analysing traces.
            skill_manager: SkillManager role for producing update operations.
            skillbook: Starting skillbook.  Creates an empty one if ``None``.
            dedup_manager: Optional deduplication manager.  Appends a
                ``DeduplicateStep`` when provided.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.  Appends a
                ``CheckpointStep`` when provided.
            checkpoint_interval: Samples between checkpoint saves.
        """
        skillbook = skillbook or Skillbook()
        steps = learning_tail(
            reflector,
            skill_manager,
            skillbook,
            dedup_manager=dedup_manager,
            dedup_interval=dedup_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    def run(
        self,
        traces: Sequence[Any],
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Analyse traces and evolve the skillbook.

        Args:
            traces: Sequence of raw trace objects (any type).
            epochs: Number of passes over all traces.
            wait: If ``True``, block until background learning completes.

        Returns:
            List of ``SampleResult``, one per trace per epoch.
        """
        return self._run(traces, epochs=epochs, wait=wait)

    def _build_context(
        self,
        raw_trace: Any,
        *,
        epoch: int,
        total_epochs: int,
        index: int,
        total: int | None,
        global_sample_index: int,
        **_: Any,
    ) -> ACEStepContext:
        """Place a raw trace directly on the context.

        No extraction, no conversion — the Reflector receives the trace
        as-is and has full freedom to analyse it.
        """
        return ACEStepContext(
            skillbook=SkillbookView(self.skillbook),
            trace=raw_trace,
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=index,
            total_steps=total,
            global_sample_index=global_sample_index,
        )
