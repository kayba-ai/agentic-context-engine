"""LangChain — runner for LangChain Runnables with ACE learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.skillbook import Skillbook
from ..integrations.langchain import LangChainExecuteStep, LangChainToTrace
from ..protocols import (
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..steps import learning_tail
from .base import ACERunner


class LangChain(ACERunner):
    """LangChain Runnable with ACE learning pipeline.

    INJECT skillbook → EXECUTE runnable → LEARN (Reflect → Tag → Update → Apply).

    Handles simple chains, AgentExecutor, and LangGraph graphs automatically.

    Usage::

        runner = LangChain.from_roles(
            runnable=my_chain,
            reflector=reflector,
            skill_manager=skill_manager,
        )
        results = runner.run([
            {"input": "What is ACE?"},
            {"input": "Explain skillbooks"},
        ])
        runner.save("chain_expert.json")
    """

    @classmethod
    def from_roles(
        cls,
        *,
        runnable: Any,
        reflector: ReflectorLike,
        skill_manager: SkillManagerLike,
        skillbook: Skillbook | None = None,
        output_parser: Optional[Callable[[Any], str]] = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
    ) -> LangChain:
        """Construct from a LangChain Runnable and pre-built role instances.

        Args:
            runnable: Any LangChain Runnable (chain, AgentExecutor, LangGraph).
            reflector: Reflector role for analysing execution traces.
            skill_manager: SkillManager role for update operations.
            skillbook: Starting skillbook.  Creates an empty one if ``None``.
            output_parser: Custom function to extract a string from runnable output.
            dedup_manager: Optional deduplication manager.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
        """
        skillbook = skillbook or Skillbook()
        steps = [
            LangChainExecuteStep(runnable, output_parser=output_parser),
            LangChainToTrace(),
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
        inputs: Sequence[Any] | Iterable[Any],
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Run inputs through the chain with learning.

        Args:
            inputs: Raw inputs (strings, dicts, message lists).
                Must be a ``Sequence`` for ``epochs > 1``.
            epochs: Number of passes over all inputs.
            wait: If ``True``, block until background learning completes.
        """
        return self._run(inputs, epochs=epochs, wait=wait)

    def _build_context(
        self,
        raw_input: Any,
        *,
        epoch: int,
        total_epochs: int,
        index: int,
        total: int | None,
        global_sample_index: int,
        **_: Any,
    ) -> ACEStepContext:
        """Place a raw input on ``ctx.sample``."""
        return ACEStepContext(
            sample=raw_input,
            skillbook=SkillbookView(self.skillbook),
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=index,
            total_steps=total,
            global_sample_index=global_sample_index,
        )
