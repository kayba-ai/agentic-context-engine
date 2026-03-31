"""TraceAnalyser — batch learning from pre-recorded execution traces."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pipeline import Pipeline
from pipeline.errors import CancellationToken
from pipeline.protocol import SampleResult, StepProtocol

from ..core.context import ACEStepContext, SkillbookView
from ..core.insight_source import TRACE_IDENTITY_METADATA_KEY, infer_trace_identity
from ..protocols import (
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..core.skillbook import Skillbook
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
    def build_steps(
        cls,
        *,
        reflector: ReflectorLike,
        skill_manager: SkillManagerLike,
        skillbook: Skillbook | None = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        extra_steps: list[StepProtocol] | None = None,
    ) -> list[StepProtocol]:
        """Return the steps that ``from_roles()`` would compose.

        Use this to inspect, modify, or extend the pipeline before
        constructing it yourself::

            steps = TraceAnalyser.build_steps(reflector=r, skill_manager=sm, ...)
            steps.append(MyCustomStep())
            pipe = Pipeline(steps)
            runner = ACERunner(pipeline=pipe, skillbook=skillbook)

        Args:
            reflector: Reflector role for analysing traces.
            skill_manager: SkillManager role for producing update operations.
            skillbook: Starting skillbook.  Creates an empty one if ``None``.
            dedup_manager: Optional deduplication manager.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
            extra_steps: Additional steps appended after the learning
                tail (e.g. ``OpikStep``).
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
        if extra_steps:
            steps.extend(extra_steps)
        return steps

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
        extra_steps: list[StepProtocol] | None = None,
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
            extra_steps: Additional steps appended after the learning
                tail (e.g. ``OpikStep``).
        """
        skillbook = skillbook or Skillbook()
        steps = cls.build_steps(
            reflector=reflector,
            skill_manager=skill_manager,
            skillbook=skillbook,
            dedup_manager=dedup_manager,
            dedup_interval=dedup_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            extra_steps=extra_steps,
        )
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    def run(
        self,
        traces: Sequence[Any],
        epochs: int = 1,
        *,
        wait: bool = True,
        cancel_token: CancellationToken | None = None,
    ) -> list[SampleResult]:
        """Analyse traces and evolve the skillbook.

        Args:
            traces: Sequence of raw trace objects (any type).
            epochs: Number of passes over all traces.
            wait: If ``True``, block until background learning completes.
            cancel_token: Optional cancellation signal.  Forwarded to
                ``Pipeline.run()`` — checked between steps and inside
                LLM calls (via contextvar).

        Returns:
            List of ``SampleResult``, one per trace per epoch.
        """
        return self._run(traces, epochs=epochs, wait=wait, cancel_token=cancel_token)

    def _build_context(  # type: ignore[override]
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
            metadata=MappingProxyType({
                TRACE_IDENTITY_METADATA_KEY: infer_trace_identity(
                    trace=raw_trace,
                    default_source_system="trace",
                ).to_dict()
            }),
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=index,
            total_steps=total,
            global_sample_index=global_sample_index,
        )
