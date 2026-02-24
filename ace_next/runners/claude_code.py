"""ClaudeCode — runner for Claude Code CLI with ACE learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.skillbook import Skillbook
from ..integrations.claude_code import ClaudeCodeExecuteStep, ClaudeCodeToTrace
from ..protocols import (
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..steps import learning_tail
from .base import ACERunner


class ClaudeCode(ACERunner):
    """Claude Code CLI with ACE learning pipeline.

    INJECT skillbook → EXECUTE Claude Code → LEARN (Reflect → Tag → Update → Apply).

    Usage::

        runner = ClaudeCode.from_roles(
            reflector=reflector,
            skill_manager=skill_manager,
            working_dir="./my_project",
        )
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
        working_dir: Optional[str] = None,
        timeout: int = 600,
        model: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
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
            working_dir: Directory where Claude Code executes.
            timeout: Execution timeout in seconds.
            model: Optional Claude model override.
            allowed_tools: Optional list of allowed tools.
            dedup_manager: Optional deduplication manager.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
        """
        skillbook = skillbook or Skillbook()
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
        """Run coding tasks with learning.

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
