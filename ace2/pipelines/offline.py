"""OfflineACE — multi-epoch runner built on the generic pipeline engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from pipeline import Pipeline, StepContext
from pipeline.protocol import SampleResult

if TYPE_CHECKING:
    from ace.deduplication import DeduplicationConfig
    from ace.roles import Agent, Reflector, SkillManager
    from ace.skillbook import Skillbook

logger = logging.getLogger(__name__)


class OfflineACE(Pipeline):
    """Multi-epoch ACE pipeline.

    Inherits from :class:`Pipeline` so it can be nested as a step
    inside another pipeline, while providing a multi-epoch ``run()``
    with checkpoint support.

    Prefer the factory methods :meth:`from_client` or :meth:`from_roles`
    for convenient construction.  Pass *steps* directly when you need a
    custom step chain (e.g. inserting a logging step between Evaluate
    and Reflect).

    Args:
        steps: Pre-built step list (or use factory methods instead).
        skillbook: The shared Skillbook that steps read and mutate.
    """

    def __init__(
        self, steps: list | None = None, *, skillbook: "Skillbook"
    ) -> None:
        super().__init__(steps=steps)
        self.skillbook = skillbook

    @classmethod
    def from_client(
        cls,
        client: Any,
        *,
        skillbook: "Optional[Skillbook]" = None,
        **kwargs: Any,
    ) -> "OfflineACE":
        """Build an OfflineACE from a single LLM client.

        Creates Agent, Reflector, and SkillManager internally.
        Extra *kwargs* are forwarded to :meth:`from_roles`.
        """
        from ace.roles import Agent, Reflector, SkillManager

        return cls.from_roles(
            agent=Agent(client),
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            skillbook=skillbook,
            **kwargs,
        )

    @classmethod
    def from_roles(
        cls,
        *,
        agent: "Agent",
        reflector: "Reflector",
        skill_manager: "SkillManager",
        skillbook: "Optional[Skillbook]" = None,
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
        dedup_config: "Optional[DeduplicationConfig]" = None,
    ) -> "OfflineACE":
        """Build an OfflineACE from individual role objects.

        Args:
            agent: Agent role instance.
            reflector: Reflector role instance.
            skill_manager: SkillManager role instance.
            skillbook: Optional skillbook (creates empty one if *None*).
            max_refinement_rounds: Passed to ReflectStep.
            reflection_window: Rolling window size for recent reflections.
            dedup_config: Optional deduplication config — installs a
                :class:`DeduplicationManager` on the skill_manager if it
                doesn't already have one.
        """
        from ace.skillbook import Skillbook as SB

        from ace2.steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep

        if skillbook is None:
            skillbook = SB()

        # Wire deduplication onto the skill_manager if requested
        if dedup_config is not None and skill_manager.dedup_manager is None:
            from ace.deduplication import DeduplicationManager

            skill_manager.dedup_manager = DeduplicationManager(dedup_config)

        steps = [
            AgentStep(agent),
            EvaluateStep(),
            ReflectStep(
                reflector,
                max_refinement_rounds=max_refinement_rounds,
                reflection_window=reflection_window,
            ),
            UpdateStep(skill_manager),
        ]
        return cls(steps=steps, skillbook=skillbook)

    # ------------------------------------------------------------------
    # Per-sample processing
    # ------------------------------------------------------------------

    def _process_one(self, ctx: StepContext) -> SampleResult:
        """Run the full step chain on *ctx*, capturing errors."""
        result = SampleResult(
            sample=ctx.sample, output=None, error=None, failed_at=None
        )
        try:
            result.output = self(ctx)
        except Exception as exc:
            result.error = exc
            result.failed_at = type(exc).__name__
            logger.warning(
                "Failed sample %d/%d epoch %d/%d: %s",
                ctx.step_index, ctx.total_steps, ctx.epoch, ctx.total_epochs, exc,
            )
        return result

    def _checkpoint(
        self, results: list[SampleResult], interval: int | None, directory: str | None
    ) -> None:
        """Save skillbook if the checkpoint interval has been reached."""
        if not interval or not directory or results[-1].error is not None:
            return
        if len(results) % interval != 0:
            return
        cp = Path(directory)
        cp.mkdir(parents=True, exist_ok=True)
        self.skillbook.save_to_file(str(cp / f"ace_checkpoint_{len(results)}.json"))
        self.skillbook.save_to_file(str(cp / "ace_latest.json"))

    # ------------------------------------------------------------------
    # Multi-epoch runner
    # ------------------------------------------------------------------

    def run(  # type: ignore[override]
        self,
        samples: Sequence[Any],
        environment: Any,
        epochs: int = 1,
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | None = None,
        wait_for_background: bool = True,
    ) -> list[SampleResult]:
        if checkpoint_interval is not None and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be provided when checkpoint_interval is set"
            )

        results: list[SampleResult] = []
        recent_reflections: tuple[str, ...] = ()
        total = len(samples)

        for epoch in range(1, epochs + 1):
            for idx, sample in enumerate(samples, start=1):
                ctx = StepContext(
                    sample=sample,
                    skillbook=self.skillbook,
                    environment=environment,
                    epoch=epoch,
                    total_epochs=epochs,
                    step_index=idx,
                    total_steps=total,
                    recent_reflections=recent_reflections,
                )
                result = self._process_one(ctx)
                if result.output is not None:
                    recent_reflections = result.output.recent_reflections
                results.append(result)
                self._checkpoint(results, checkpoint_interval, checkpoint_dir)

        if wait_for_background:
            self.wait_for_background()

        return results
