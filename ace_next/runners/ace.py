"""ACE — full adaptive pipeline runner."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pipeline import Pipeline
from pipeline.protocol import SampleResult, StepProtocol

from ..core.context import ACEStepContext, SkillbookView
from ..core.environments import Sample, TaskEnvironment
from ..protocols import (
    AgentLike,
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..core.skillbook import Skillbook
from ..steps import AgentStep, EvaluateStep, learning_tail
from .base import ACERunner


class ACE(ACERunner):
    """Live adaptive pipeline: Agent -> Evaluate -> Reflect -> Tag -> Update -> Apply.

    The full ACE loop.  An agent executes, the environment evaluates, the
    reflector analyses, and the skill manager updates the skillbook.

    A single class handles both single-pass (``epochs=1``) and multi-epoch
    batch training (``epochs > 1``).

    Use when you are building a new agent from scratch and want
    closed-loop learning where the agent improves in real time.
    """

    @classmethod
    def from_roles(
        cls,
        *,
        agent: AgentLike,
        reflector: ReflectorLike,
        skill_manager: SkillManagerLike,
        environment: TaskEnvironment | None = None,
        skillbook: Skillbook | None = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        extra_steps: list[StepProtocol] | None = None,
    ) -> ACE:
        """Construct from pre-built role instances.

        Args:
            agent: Agent role for producing answers.
            reflector: Reflector role for analysing execution.
            skill_manager: SkillManager role for update operations.
            environment: Optional task environment for evaluation feedback.
                When provided, ``EvaluateStep`` generates feedback that
                enriches the trace.  When omitted, the trace still contains
                the agent's output, question, context, and ground truth.
            skillbook: Starting skillbook.  Creates an empty one if ``None``.
            dedup_manager: Optional deduplication manager.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
            extra_steps: Additional steps appended after the learning
                tail (e.g. ``OpikStep``).
        """
        skillbook = skillbook or Skillbook()
        steps = [
            AgentStep(agent),
            EvaluateStep(environment),
            *learning_tail(
                reflector,
                skill_manager,
                skillbook,
                dedup_manager=dedup_manager,
                dedup_interval=dedup_interval,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                extra_steps=extra_steps,
            ),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    def run(
        self,
        samples: Sequence[Sample] | Iterable[Sample],
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Run the adaptive pipeline over samples.

        Args:
            samples: Input samples.  Must be a ``Sequence`` for
                ``epochs > 1``.  ``Iterable`` is accepted when
                ``epochs=1`` (consumed once).
            epochs: Number of passes over the samples.
            wait: If ``True``, block until background learning completes.

        Returns:
            List of ``SampleResult``, one per sample per epoch.

        Raises:
            ValueError: If ``epochs > 1`` and *samples* is not a
                ``Sequence``.
        """
        return self._run(samples, epochs=epochs, wait=wait)

    def _build_context(
        self,
        sample: Sample,
        *,
        epoch: int,
        total_epochs: int,
        index: int,
        total: int | None,
        global_sample_index: int,
        **_: Any,
    ) -> ACEStepContext:
        """Map a ``Sample`` to an ``ACEStepContext`` for the full pipeline.

        Sets ``sample`` and ``skillbook`` on the context.  The environment
        (if any) is injected into ``EvaluateStep`` at construction time —
        it does not appear on the context.
        """
        return ACEStepContext(
            sample=sample,
            skillbook=SkillbookView(self.skillbook),
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=index,
            total_steps=total,
            global_sample_index=global_sample_index,
        )
