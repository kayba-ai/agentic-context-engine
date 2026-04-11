"""ACELiteLLM — batteries-included conversational agent with ACE learning."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

from pydantic_ai.settings import ModelSettings

from pipeline.protocol import SampleResult

from ..core.context import ACEStepContext, SkillbookView
from ..core.environments import Sample, TaskEnvironment
from ..core.outputs import AgentOutput
from ..core.skillbook import Skillbook
from ..implementations import Agent, Reflector, SkillManager
from ..integrations import wrap_skillbook_context
from ..protocols import (
    AgentLike,
    DeduplicationConfig,
    DeduplicationManagerLike,
    ReflectorLike,
    SkillManagerLike,
)
from ..providers.pydantic_ai import resolve_model, settings_from_config
from ..steps import learning_tail
from .ace import ACE
from .trace_analyser import TraceAnalyser

logger = logging.getLogger(__name__)


class ACELiteLLM:
    """PydanticAI-powered conversational agent with ACE learning.

    Bundles Agent, Reflector, SkillManager, and Skillbook into a simple
    interface.  Delegates to :class:`ACE` for batch learning and
    :class:`TraceAnalyser` for trace-based learning.

    Two construction paths:

    1. ``ACELiteLLM("gpt-4o-mini", ...)`` — builds PydanticAI-backed
       roles from a model string.
    2. ``ACELiteLLM.from_model("gpt-4o-mini", ...)`` — explicit factory
       with full parameter control.

    Example::

        ace = ACELiteLLM.from_model("gpt-4o-mini")
        answer = ace.ask("What is 2+2?")
        ace.learn(samples, environment=SimpleEnvironment(), epochs=3)
        ace.save("learned.json")
    """

    def __init__(
        self,
        model: str,
        *,
        skillbook: Skillbook | None = None,
        skillbook_path: str | None = None,
        environment: TaskEnvironment | None = None,
        agent: AgentLike | None = None,
        reflector: ReflectorLike | None = None,
        skill_manager: SkillManagerLike | None = None,
        model_settings: ModelSettings | None = None,
        dedup_config: DeduplicationConfig | None = None,
        dedup_manager: DeduplicationManagerLike | None = None,
        dedup_interval: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        is_learning: bool = True,
        logfire: bool = False,
    ) -> None:
        # Resolve skillbook
        if skillbook_path:
            self._skillbook = Skillbook.load_from_file(skillbook_path)
        elif skillbook is not None:
            self._skillbook = skillbook
        else:
            self._skillbook = Skillbook()

        # Build roles (use provided or create PydanticAI-backed defaults)
        self.agent: AgentLike = agent or Agent(model, model_settings=model_settings)
        self.reflector: ReflectorLike = reflector or Reflector(
            model, model_settings=model_settings
        )
        self.skill_manager: SkillManagerLike = skill_manager or SkillManager(
            model, model_settings=model_settings
        )

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

        # Logfire observability (explicit opt-in — fail loudly)
        if logfire:
            from ..observability import configure_logfire

            if not configure_logfire():
                raise ImportError(
                    "logfire=True requires the 'logfire' package. "
                    "Install it with: pip install ace-framework[logfire]"
                )

        # Lazy-init caches
        self._ace: ACE | None = None
        self._analyser: TraceAnalyser | None = None

        # Last interaction for learn_from_feedback()
        self._last_interaction: tuple[str, AgentOutput] | None = None

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_setup(
        cls,
        *,
        config_dir: str | Path | None = None,
        validate: bool = False,
        **kwargs: Any,
    ) -> ACELiteLLM:
        """Build from ace.toml + .env (created by ``ace setup``).

        Looks for ``ace.toml`` in the current directory and parents.
        Loads ``.env`` for API keys. Optionally validates each model
        connection before proceeding.

        Args:
            config_dir: Explicit directory containing ace.toml.
                If None, searches current directory and parents.
            validate: If True, run a test LLM call for each configured
                model before building. Raises on failure.
            **kwargs: Extra kwargs forwarded to the constructor
                (skillbook, environment, etc.).

        Raises:
            FileNotFoundError: If no ace.toml is found.
            ConnectionError: If validate=True and a model fails.
        """
        from ..providers.config import (
            find_config,
            load_config,
            load_dotenv,
        )

        # Load .env first so keys are available
        load_dotenv()

        # Find and load ace.toml
        if config_dir is not None:
            config = load_config(config_dir)
        else:
            config_path = find_config()
            if config_path is None:
                raise FileNotFoundError(
                    "No ace.toml found. Run `ace setup` to create one, "
                    "or use ACELiteLLM.from_model() / ACELiteLLM.from_config()."
                )
            config = load_config(config_path.parent)

        return cls.from_config(config, validate=validate, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Any,  # ACEModelConfig — Any to avoid circular import at module level
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> ACELiteLLM:
        """Build from an ``ACEModelConfig`` with per-role model selection.

        API keys are resolved from the environment (not from config).
        Each role gets its own PydanticAI-backed implementation,
        allowing different models for Agent, Reflector, and SkillManager.

        Args:
            config: An ``ACEModelConfig`` mapping roles to models.
            validate: If True, validate each model connection first.
            **kwargs: Extra kwargs forwarded to the constructor
                (skillbook, environment, etc.).

        Example::

            from ace.providers.config import ACEModelConfig, ModelConfig

            config = ACEModelConfig(
                default=ModelConfig(model="gpt-4o-mini"),
                agent=ModelConfig(model="claude-sonnet-4-20250514"),
            )
            ace = ACELiteLLM.from_config(config)
        """
        from ..providers.config import ACEModelConfig

        if not isinstance(config, ACEModelConfig):
            raise TypeError(f"Expected ACEModelConfig, got {type(config).__name__}")

        if validate:
            from ..providers.registry import validate_connection

            seen: set[str] = set()
            for role in ("agent", "reflector", "skill_manager"):
                mc = config.for_role(role)
                if mc.model in seen:
                    continue
                seen.add(mc.model)
                result = validate_connection(mc.model)
                if not result.success:
                    raise ConnectionError(
                        f"Model '{mc.model}' (for {role}) failed validation: "
                        f"{result.error}"
                    )

        agent_config = config.for_role("agent")
        reflector_config = config.for_role("reflector")
        sm_config = config.for_role("skill_manager")

        return cls(
            agent_config.model,
            agent=Agent(
                agent_config.model,
                model_settings=settings_from_config(agent_config),
            ),
            reflector=Reflector(
                reflector_config.model,
                model_settings=settings_from_config(reflector_config),
            ),
            skill_manager=SkillManager(
                sm_config.model,
                model_settings=settings_from_config(sm_config),
            ),
            **kwargs,
        )

    @classmethod
    def from_model(
        cls,
        model: str = "gpt-4o-mini",
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        skillbook: Skillbook | None = None,
        skillbook_path: Optional[str] = None,
        environment: Optional[TaskEnvironment] = None,
        dedup_config: Optional[DeduplicationConfig] = None,
        dedup_interval: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 10,
        is_learning: bool = True,
        logfire: bool = False,
    ) -> ACELiteLLM:
        """Build from a model string.

        Args:
            model: LiteLLM model identifier (e.g. ``"gpt-4o-mini"``).
            max_tokens: Max tokens for LLM responses.
            temperature: Sampling temperature.
            skillbook: Starting skillbook.
            skillbook_path: Path to load skillbook from.
            environment: Task environment for evaluation.
            dedup_config: Deduplication configuration.
            dedup_interval: Samples between deduplication runs.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Samples between checkpoint saves.
            is_learning: Whether learning is enabled.
            logfire: Enable Logfire observability (auto-instruments PydanticAI).
        """
        model_settings = ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return cls(
            model,
            model_settings=model_settings,
            skillbook=skillbook,
            skillbook_path=skillbook_path,
            environment=environment,
            dedup_config=dedup_config,
            dedup_interval=dedup_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            is_learning=is_learning,
            logfire=logfire,
        )

    # ------------------------------------------------------------------
    # Lazy-init runners
    # ------------------------------------------------------------------

    def _get_extra_steps(self) -> list[Any] | None:
        """Return extra pipeline steps or None."""
        return None

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
                extra_steps=self._get_extra_steps(),
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
                extra_steps=self._get_extra_steps(),
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

        Runs the standard ``learning_tail`` pipeline (ReflectStep,
        UpdateStep, ApplyStep) on the most recent ``ask()``
        call with the provided feedback.

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

        # Build synthetic trace (same format EvaluateStep produces)
        trace = {
            "question": question,
            "context": "",
            "ground_truth": ground_truth,
            "reasoning": agent_output.reasoning,
            "answer": agent_output.final_answer,
            "skill_ids": agent_output.skill_ids,
            "feedback": feedback,
        }

        ctx = ACEStepContext(
            skillbook=SkillbookView(self._skillbook),
            trace=trace,
        )

        # Same learning pipeline as learn() and learn_from_traces()
        from pipeline import Pipeline

        steps = learning_tail(
            self.reflector,
            self.skill_manager,
            self._skillbook,
        )
        pipe = Pipeline(steps)
        results = pipe.run([ctx])
        pipe.wait_for_background()  # ReflectStep has async_boundary=True
        return len(results) > 0 and results[0].error is None

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
