"""OnlineACE — single-pass stream runner built on the generic pipeline engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Optional

from pipeline import Pipeline, StepContext
from pipeline.protocol import SampleResult

if TYPE_CHECKING:
    from ace.deduplication import DeduplicationConfig
    from ace.roles import Agent, Reflector, SkillManager
    from ace.skillbook import Skillbook

logger = logging.getLogger(__name__)


class OnlineACE:
    """Single-pass ACE runner for continuous / streaming adaptation.

    Thin wrapper that iterates once over an arbitrary iterable,
    builds a :class:`StepContext` per sample, and delegates to the pipeline.

    Args:
        pipeline: A wired Pipeline (use :func:`ace_pipeline` to build one).
        skillbook: The shared Skillbook that steps read and mutate.
    """

    def __init__(self, pipeline: Pipeline, skillbook: "Skillbook") -> None:
        self.pipeline = pipeline
        self.skillbook = skillbook

    @classmethod
    def from_client(
        cls,
        client: Any,
        *,
        skillbook: "Optional[Skillbook]" = None,
        **kwargs: Any,
    ) -> "OnlineACE":
        """Build an OnlineACE from a single LLM client.

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
    ) -> "OnlineACE":
        """Build an OnlineACE from individual role objects.

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

        if skillbook is None:
            skillbook = SB()

        if dedup_config is not None and skill_manager.dedup_manager is None:
            from ace.deduplication import DeduplicationManager

            skill_manager.dedup_manager = DeduplicationManager(dedup_config)

        from ace2.pipelines import ace_pipeline

        pipe = ace_pipeline(
            agent,
            reflector,
            skill_manager,
            max_refinement_rounds=max_refinement_rounds,
            reflection_window=reflection_window,
        )
        return cls(pipe, skillbook)

    def run(
        self,
        samples: Iterable[Any],
        environment: Any,
        wait_for_background: bool = True,
    ) -> list[SampleResult]:
        results: list[SampleResult] = []
        recent_reflections: tuple[str, ...] = ()

        for step_idx, sample in enumerate(samples, start=1):
            ctx = StepContext(
                sample=sample,
                skillbook=self.skillbook,
                environment=environment,
                epoch=1,
                total_epochs=1,
                step_index=step_idx,
                total_steps=step_idx,
                recent_reflections=recent_reflections,
            )

            result = SampleResult(
                sample=sample, output=None, error=None, failed_at=None
            )

            try:
                out_ctx = self.pipeline(ctx)
                result.output = out_ctx
                recent_reflections = out_ctx.recent_reflections
            except Exception as exc:
                result.error = exc
                result.failed_at = type(exc).__name__
                logger.warning("Failed sample %d: %s", step_idx, exc)

            results.append(result)

        if wait_for_background:
            self.pipeline.wait_for_background()

        return results
