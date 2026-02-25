"""LangChain — LangChain Runnable with ACE learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

from pipeline import Pipeline
from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.skillbook import Skillbook
from ..integrations import wrap_skillbook_context
from ..integrations.langchain import LangChainExecuteStep, LangChainToTrace
from ..protocols import (
    DeduplicationConfig,
    DeduplicationManagerLike,
    LLMClientLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..steps import learning_tail
from .base import ACERunner


class LangChain(ACERunner):
    """LangChain Runnable with ACE learning pipeline.

    INJECT skillbook -> EXECUTE runnable -> LEARN (Reflect -> Tag -> Update -> Apply).

    Handles simple chains, AgentExecutor, and LangGraph graphs automatically.

    Two construction paths:

    1. ``LangChain.from_roles(runnable, reflector, skill_manager, ...)``
       — pre-built roles.
    2. ``LangChain.from_model(runnable, ace_model="gpt-4o-mini", ...)``
       — builds ACE roles from a model string.

    Example::

        runner = LangChain.from_model(my_chain, ace_model="gpt-4o-mini")
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
        skillbook_path: Optional[str] = None,
        output_parser: Optional[Callable[[Any], str]] = None,
        dedup_config: Optional[DeduplicationConfig] = None,
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
            skillbook_path: Path to load skillbook from.
            output_parser: Custom function to extract a string from runnable output.
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
            LangChainExecuteStep(runnable, output_parser=output_parser),
            LangChainToTrace(),
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
        runnable: Any,
        *,
        ace_model: str = "gpt-4o-mini",
        ace_max_tokens: int = 2048,
        ace_llm: Optional[LLMClientLike] = None,
        **kwargs: Any,
    ) -> LangChain:
        """Build ACE roles from a model string.

        Args:
            runnable: Any LangChain Runnable (chain, AgentExecutor, LangGraph).
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
            runnable=runnable,
            reflector=Reflector(ace_llm),
            skill_manager=SkillManager(ace_llm),
            **kwargs,
        )

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

    def invoke(self, input: Any, **kwargs: Any) -> list[SampleResult]:
        """Single-input convenience — wraps in a list and delegates to :meth:`run`.

        Args:
            input: A single chain input.
            **kwargs: Forwarded to :meth:`run`.
        """
        return self.run([input], **kwargs)

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
    load_skillbook = ACERunner.load
    wait_for_learning = ACERunner.wait_for_background
