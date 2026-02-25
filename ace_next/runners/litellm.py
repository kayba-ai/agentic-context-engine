"""ACELiteLLM — batteries-included conversational agent with ACE learning."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pipeline.protocol import SampleResult

from ..core.environments import Sample, TaskEnvironment
from ..core.outputs import AgentOutput
from ..core.skillbook import Skillbook
from ..implementations import Agent, Reflector, SkillManager
from ..integrations import wrap_skillbook_context
from ..protocols import (
    AgentLike,
    DeduplicationConfig,
    DeduplicationManagerLike,
    LLMClientLike,
    ReflectorLike,
    SkillManagerLike,
)
from .ace import ACE
from .trace_analyser import TraceAnalyser

logger = logging.getLogger(__name__)


class ACELiteLLM:
    """LiteLLM-powered conversational agent with ACE learning.

    Bundles Agent, Reflector, SkillManager, and Skillbook into a simple
    interface.  Delegates to :class:`ACE` for batch learning and
    :class:`TraceAnalyser` for trace-based learning.

    Two construction paths:

    1. ``ACELiteLLM(llm, ...)`` — accepts a pre-built ``LLMClientLike``
       and optionally pre-built role instances.
    2. ``ACELiteLLM.from_model("gpt-4o-mini", ...)`` — builds a
       ``LiteLLMClient``, wraps with Instructor, and creates all roles
       automatically.

    Example::

        ace = ACELiteLLM.from_model("gpt-4o-mini")
        answer = ace.ask("What is 2+2?")
        ace.learn(samples, environment=SimpleEnvironment(), epochs=3)
        ace.save("learned.json")
    """

    def __init__(
        self,
        llm: LLMClientLike,
        *,
        skillbook: Skillbook | None = None,
        skillbook_path: str | None = None,
        environment: TaskEnvironment | None = None,
        agent: AgentLike | None = None,
        reflector: ReflectorLike | None = None,
        skill_manager: SkillManagerLike | None = None,
        dedup_config: DeduplicationConfig | None = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        is_learning: bool = True,
    ) -> None:
        # Resolve skillbook
        if skillbook_path:
            self._skillbook = Skillbook.load_from_file(skillbook_path)
        elif skillbook is not None:
            self._skillbook = skillbook
        else:
            self._skillbook = Skillbook()

        # Build roles (use provided or create defaults)
        self.agent: AgentLike = agent or Agent(llm)
        self.reflector: ReflectorLike = reflector or Reflector(llm)
        self.skill_manager: SkillManagerLike = skill_manager or SkillManager(llm)

        self.environment = environment
        self.is_learning = is_learning

        # Resolve dedup manager
        if dedup_manager is not None:
            self._dedup_manager: DeduplicationManagerLike | None = dedup_manager
        elif dedup_config is not None:
            from ..deduplication import DeduplicationManager

            self._dedup_manager = DeduplicationManager(dedup_config)
        else:
            self._dedup_manager = None

        self._dedup_interval = dedup_interval
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_interval = checkpoint_interval

        # Lazy-init caches
        self._ace: ACE | None = None
        self._analyser: TraceAnalyser | None = None

        # Last interaction for learn_from_feedback()
        self._last_interaction: tuple[str, AgentOutput] | None = None

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: str = "gpt-4o-mini",
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        ssl_verify: Optional[Union[bool, str]] = None,
        skillbook: Skillbook | None = None,
        skillbook_path: Optional[str] = None,
        environment: Optional[TaskEnvironment] = None,
        dedup_config: Optional[DeduplicationConfig] = None,
        dedup_interval: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 10,
        is_learning: bool = True,
        **llm_kwargs: Any,
    ) -> ACELiteLLM:
        """Build from a model string (creates LiteLLMClient + roles).

        Args:
            model: LiteLLM model identifier (e.g. ``"gpt-4o-mini"``).
            max_tokens: Max tokens for LLM responses.
            temperature: Sampling temperature.
            api_key: API key (or set via environment variable).
            base_url: Custom API base URL.
            extra_headers: Extra HTTP headers for the LLM provider.
            ssl_verify: SSL verification setting.
            skillbook: Starting skillbook.
            skillbook_path: Path to load skillbook from.
            environment: Task environment for evaluation.
            dedup_config: Deduplication configuration.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
            is_learning: Whether learning is enabled.
            **llm_kwargs: Extra kwargs forwarded to ``LiteLLMClient``.
        """
        from ..providers import LiteLLMClient

        llm = LiteLLMClient(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=base_url,
            **llm_kwargs,
        )
        return cls(
            llm,
            skillbook=skillbook,
            skillbook_path=skillbook_path,
            environment=environment,
            dedup_config=dedup_config,
            dedup_interval=dedup_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            is_learning=is_learning,
        )

    # ------------------------------------------------------------------
    # Lazy-init runners
    # ------------------------------------------------------------------

    def _get_ace(self, environment: TaskEnvironment | None = None) -> ACE:
        """Return (or build) the cached ACE runner."""
        env = environment or self.environment
        # Invalidate if environment changed
        if self._ace is not None and env is not self.environment:
            self._ace = None
            self.environment = env
        if self._ace is None:
            self._ace = ACE.from_roles(
                agent=self.agent,
                reflector=self.reflector,
                skill_manager=self.skill_manager,
                environment=env,
                skillbook=self._skillbook,
                dedup_manager=self._dedup_manager,
                dedup_interval=self._dedup_interval,
                checkpoint_dir=self._checkpoint_dir,
                checkpoint_interval=self._checkpoint_interval,
            )
        return self._ace

    def _get_analyser(self) -> TraceAnalyser:
        """Return (or build) the cached TraceAnalyser."""
        if self._analyser is None:
            self._analyser = TraceAnalyser.from_roles(
                reflector=self.reflector,
                skill_manager=self.skill_manager,
                skillbook=self._skillbook,
                dedup_manager=self._dedup_manager,
                dedup_interval=self._dedup_interval,
                checkpoint_dir=self._checkpoint_dir,
                checkpoint_interval=self._checkpoint_interval,
            )
        return self._analyser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str, context: str = "") -> str:
        """Ask a question using the current skillbook.

        Direct Agent call — does not go through the pipeline.  Stores
        the interaction for optional :meth:`learn_from_feedback`.

        Args:
            question: The question to answer.
            context: Optional context for the question.

        Returns:
            The agent's final answer.
        """
        output = self.agent.generate(
            question=question,
            context=context,
            skillbook=self._skillbook,
        )
        self._last_interaction = (question, output)
        return output.final_answer

    def learn(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment | None = None,
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Run the full ACE learning pipeline over samples.

        Args:
            samples: Training samples with questions and ground truth.
            environment: Task environment for evaluation.  Falls back
                to the environment set at construction time.
            epochs: Number of passes over the samples.
            wait: If ``True``, block until background learning completes.

        Returns:
            List of ``SampleResult``, one per sample per epoch.

        Raises:
            RuntimeError: If learning is disabled.
        """
        if not self.is_learning:
            raise RuntimeError("Learning is disabled. Call enable_learning() first.")
        return self._get_ace(environment).run(samples, epochs=epochs, wait=wait)

    def learn_from_traces(
        self,
        traces: Sequence[Any],
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]:
        """Learn from pre-recorded execution traces.

        Args:
            traces: Raw trace objects (dicts, framework results, etc.).
            epochs: Number of passes over the traces.
            wait: If ``True``, block until background learning completes.

        Returns:
            List of ``SampleResult``, one per trace per epoch.

        Raises:
            RuntimeError: If learning is disabled.
        """
        if not self.is_learning:
            raise RuntimeError("Learning is disabled. Call enable_learning() first.")
        return self._get_analyser().run(traces, epochs=epochs, wait=wait)

    def learn_from_feedback(
        self,
        feedback: str,
        ground_truth: str | None = None,
    ) -> bool:
        """Learn from the last :meth:`ask` interaction.

        Runs the Reflector and SkillManager directly (no pipeline) on
        the most recent ``ask()`` call with the provided feedback.

        Args:
            feedback: User feedback about the answer quality.
            ground_truth: Optional correct answer.

        Returns:
            ``True`` if learning was applied, ``False`` if no prior
            interaction exists or learning is disabled.
        """
        if not self.is_learning or self._last_interaction is None:
            return False

        question, agent_output = self._last_interaction

        reflection = self.reflector.reflect(
            question=question,
            agent_output=agent_output,
            skillbook=self._skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
        )
        sm_output = self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self._skillbook,
            question_context=f"User interaction: {question}",
            progress="Learning from user feedback",
        )
        self._skillbook.apply_update(sm_output.update)
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def skillbook(self) -> Skillbook:
        """The current skillbook."""
        return self._skillbook

    def save(self, path: str) -> None:
        """Save the skillbook to disk."""
        self._skillbook.save_to_file(path)

    def load(self, path: str) -> None:
        """Load a skillbook from disk.

        Invalidates cached runners (they hold stale skillbook refs).
        """
        self._skillbook = Skillbook.load_from_file(path)
        self._ace = None
        self._analyser = None

    def enable_learning(self) -> None:
        """Enable learning."""
        self.is_learning = True

    def disable_learning(self) -> None:
        """Disable learning."""
        self.is_learning = False

    def get_strategies(self) -> str:
        """Return formatted skillbook strategies for display."""
        if not self._skillbook.skills():
            return ""
        return wrap_skillbook_context(self._skillbook)

    def wait_for_background(self, timeout: float | None = None) -> None:
        """Block until all background learning completes."""
        if self._ace is not None:
            self._ace.wait_for_background(timeout)
        if self._analyser is not None:
            self._analyser.wait_for_background(timeout)

    @property
    def learning_stats(self) -> dict[str, int]:
        """Return background learning progress."""
        stats: dict[str, int] = {}
        if self._ace is not None:
            stats.update(self._ace.learning_stats)
        if self._analyser is not None:
            stats.update(self._analyser.learning_stats)
        return stats

    # Backward-compat aliases
    save_skillbook = save
    load_skillbook = load
    wait_for_learning = wait_for_background
