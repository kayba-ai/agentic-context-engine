"""BrowserUse — browser-use agent with ACE learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.skillbook import Skillbook
from ..integrations import wrap_skillbook_context
from ..integrations.browser_use import BrowserExecuteStep, BrowserToTrace
from ..protocols import (
    DeduplicationConfig,
    DeduplicationManagerLike,
    LLMClientLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..steps import learning_tail
from .base import ACERunner


class BrowserUse(ACERunner):
    """Browser-use agent with ACE learning pipeline.

    INJECT skillbook -> EXECUTE browser-use -> LEARN (Reflect -> Tag -> Update -> Apply).

    Two construction paths:

    1. ``BrowserUse.from_roles(browser_llm, reflector, skill_manager, ...)``
       — pre-built roles.
    2. ``BrowserUse.from_model(browser_llm, ace_model="gpt-4o-mini", ...)``
       — builds ACE roles from a model string.

    Example::

        runner = BrowserUse.from_model(
            browser_llm=ChatOpenAI(model="gpt-4o"),
            ace_model="gpt-4o-mini",
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
        skillbook_path: Optional[str] = None,
        browser: Any = None,
        agent_kwargs: dict[str, Any] | None = None,
        dedup_config: Optional[DeduplicationConfig] = None,
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
            skillbook_path: Path to load skillbook from.
            browser: Optional browser-use Browser instance.
            agent_kwargs: Extra kwargs forwarded to browser-use Agent.
            dedup_config: Deduplication configuration.
            dedup_manager: Optional pre-built deduplication manager.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
        """
        # Resolve skillbook
        if skillbook_path:
            skillbook = Skillbook.load_from_file(skillbook_path)
        elif skillbook is None:
            skillbook = Skillbook()

        # Resolve dedup manager
        dm = dedup_manager
        if dm is None and dedup_config is not None:
            from ..deduplication import DeduplicationManager

            dm = DeduplicationManager(dedup_config)

        steps = [
            BrowserExecuteStep(browser_llm, browser=browser, **(agent_kwargs or {})),
            BrowserToTrace(),
            *learning_tail(
                reflector,
                skill_manager,
                skillbook,
                dedup_manager=dm,
                dedup_interval=dedup_interval,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
            ),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    @classmethod
    def from_model(
        cls,
        browser_llm: Any,
        *,
        ace_model: str = "gpt-4o-mini",
        ace_max_tokens: int = 2048,
        ace_llm: Optional[LLMClientLike] = None,
        **kwargs: Any,
    ) -> BrowserUse:
        """Build ACE roles from a model string.

        Args:
            browser_llm: LLM for browser-use execution.
            ace_model: Model identifier for ACE roles.
            ace_max_tokens: Max tokens for ACE LLM.
            ace_llm: Optional pre-built LLM for ACE roles.
            **kwargs: Forwarded to :meth:`from_roles`.
        """
        from ..implementations import Reflector, SkillManager

        if ace_llm is None:
            from ..providers import LiteLLMClient

            ace_llm = LiteLLMClient(model=ace_model, max_tokens=ace_max_tokens)

        return cls.from_roles(
            browser_llm=browser_llm,
            reflector=Reflector(ace_llm),
            skill_manager=SkillManager(ace_llm),
            **kwargs,
        )

    def run(
        self,
        tasks: Sequence[str] | Iterable[str] | str,
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Run browser tasks with learning.

        Args:
            tasks: Single task string or list of task strings.
                Must be a ``Sequence`` for ``epochs > 1``.
            epochs: Number of passes over all tasks.
            wait: If ``True``, block until background learning completes.
        """
        if isinstance(tasks, str):
            tasks = [tasks]
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

    # ------------------------------------------------------------------
    # Convenience lifecycle methods
    # ------------------------------------------------------------------

    def get_strategies(self) -> str:
        """Return formatted skillbook strategies for display."""
        if not self.skillbook.skills():
            return ""
        return wrap_skillbook_context(self.skillbook)

    # Backward-compat aliases
    save_skillbook = ACERunner.save
    load_skillbook = save_skillbook  # alias for discoverability
    wait_for_learning = ACERunner.wait_for_background
