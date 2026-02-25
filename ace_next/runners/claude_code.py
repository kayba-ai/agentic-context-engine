"""ClaudeCode — Claude Code CLI runner with ACE learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.skillbook import Skillbook
from ..integrations import wrap_skillbook_context
from ..integrations.claude_code import ClaudeCodeExecuteStep, ClaudeCodeToTrace
from ..protocols import (
    DeduplicationConfig,
    DeduplicationManagerLike,
    LLMClientLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..steps import learning_tail
from .base import ACERunner


class ClaudeCode(ACERunner):
    """Claude Code CLI with ACE learning pipeline.

    INJECT skillbook → EXECUTE Claude Code → LEARN (Reflect → Tag → Update → Apply).

    Two construction paths:

    1. ``ClaudeCode.from_roles(reflector, skill_manager, working_dir=..., ...)``
       — pre-built roles.
    2. ``ClaudeCode.from_model(working_dir=..., ace_model="gpt-4o-mini", ...)``
       — builds ACE roles from a model string.

    Example::

        runner = ClaudeCode.from_model(working_dir="./my_project")
        results = runner.run([
            "Add unit tests for utils.py",
            "Refactor the auth module",
        ])
        runner.save("code_expert.json")
    """

    @classmethod
    def from_roles(
        cls,
        *,
        reflector: ReflectorLike,
        skill_manager: SkillManagerLike,
        skillbook: Skillbook | None = None,
        skillbook_path: Optional[str] = None,
        working_dir: Optional[str] = None,
        timeout: int = 600,
        model: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        dedup_config: Optional[DeduplicationConfig] = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
    ) -> ClaudeCode:
        """Construct from pre-built role instances.

        Args:
            reflector: Reflector role for analysing execution traces.
            skill_manager: SkillManager role for update operations.
            skillbook: Starting skillbook.  Creates an empty one if ``None``.
            skillbook_path: Path to load skillbook from.
            working_dir: Directory where Claude Code executes.
            timeout: Execution timeout in seconds.
            model: Optional Claude model override.
            allowed_tools: Optional list of allowed tools.
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
            ClaudeCodeExecuteStep(
                working_dir=working_dir,
                timeout=timeout,
                model=model,
                allowed_tools=allowed_tools,
            ),
            ClaudeCodeToTrace(),
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
        *,
        working_dir: Optional[str] = None,
        ace_model: str = "gpt-4o-mini",
        ace_max_tokens: int = 2048,
        ace_llm: Optional[LLMClientLike] = None,
        **kwargs: Any,
    ) -> ClaudeCode:
        """Build ACE roles from a model string.

        Args:
            working_dir: Directory where Claude Code executes.
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
            reflector=Reflector(ace_llm),
            skill_manager=SkillManager(ace_llm),
            working_dir=working_dir,
            **kwargs,
        )

    def run(
        self,
        tasks: Sequence[str] | Iterable[str] | str,
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Run coding tasks with learning.

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
    load_skillbook = save_skillbook
    wait_for_learning = ACERunner.wait_for_background
