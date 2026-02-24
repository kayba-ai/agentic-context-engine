"""BrowserUse — runner for browser-use agents with ACE learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.skillbook import Skillbook
from ..integrations.browser_use import BrowserExecuteStep, BrowserToTrace
from ..protocols import (
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..steps import learning_tail
from .base import ACERunner


class BrowserUse(ACERunner):
    """Browser-use agent with ACE learning pipeline.

    INJECT skillbook → EXECUTE browser-use → LEARN (Reflect → Tag → Update → Apply).

    Usage::

        runner = BrowserUse.from_roles(
            browser_llm=ChatOpenAI(model="gpt-4o"),
            reflector=reflector,
            skill_manager=skill_manager,
        )
        results = runner.run(["Find top HN post", "Check weather in NYC"])
        runner.save("browser_expert.json")
    """

    @classmethod
    def from_roles(
        cls,
        *,
        browser_llm: Any,
        reflector: ReflectorLike,
        skill_manager: SkillManagerLike,
        skillbook: Skillbook | None = None,
        browser: Any = None,
        agent_kwargs: dict[str, Any] | None = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
    ) -> BrowserUse:
        """Construct from pre-built role instances.

        Args:
            browser_llm: LLM for browser-use execution.
            reflector: Reflector role for analysing execution traces.
            skill_manager: SkillManager role for update operations.
            skillbook: Starting skillbook.  Creates an empty one if ``None``.
            browser: Optional browser-use Browser instance.
            agent_kwargs: Extra kwargs forwarded to browser-use Agent.
            dedup_manager: Optional deduplication manager.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
        """
        skillbook = skillbook or Skillbook()
        steps = [
            BrowserExecuteStep(browser_llm, browser=browser, **(agent_kwargs or {})),
            BrowserToTrace(),
            *learning_tail(
                reflector,
                skill_manager,
                skillbook,
                dedup_manager=dedup_manager,
                dedup_interval=dedup_interval,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
            ),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    def run(
        self,
        tasks: Sequence[str] | Iterable[str],
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Run browser tasks with learning.

        Args:
            tasks: Task strings.  Must be a ``Sequence`` for ``epochs > 1``.
            epochs: Number of passes over all tasks.
            wait: If ``True``, block until background learning completes.
        """
        return self._run(tasks, epochs=epochs, wait=wait)

    def _build_context(
        self,
        task: str,
        *,
        epoch: int,
        total_epochs: int,
        index: int,
        total: int | None,
        global_sample_index: int,
        **_: Any,
    ) -> ACEStepContext:
        """Place a raw task string on ``ctx.sample``."""
        return ACEStepContext(
            sample=task,
            skillbook=SkillbookView(self.skillbook),
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=index,
            total_steps=total,
            global_sample_index=global_sample_index,
        )
