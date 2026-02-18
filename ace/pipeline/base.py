"""ACE pipeline base — core interfaces and shared orchestration logic."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ..async_learning import AsyncLearningPipeline
    from ..deduplication import DeduplicationConfig
    from ..observability.opik_integration import OpikIntegration

from ..skillbook import Skillbook
from ..roles import (
    Agent,
    AgentOutput,
    Reflector,
    ReflectorOutput,
    SkillManager,
    SkillManagerOutput,
)
from ..environments import (
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    ACEStepResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step context — state flowing through the pipeline
# ---------------------------------------------------------------------------


@dataclass
class StepContext:
    """Mutable state passed through each pipeline step for a single sample.

    Each step reads what it needs and writes its output back before returning.
    The pipeline creates a fresh context per sample and converts it to an
    ``ACEStepResult`` when all steps complete.

    Attributes:
        sample: The input sample being processed.
        skillbook: Shared skillbook (steps may modify it in-place).
        environment: Task environment used for evaluation.
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs in this run.
        step_index: Current sample index within the epoch (1-indexed).
        total_steps: Total samples in the epoch.
        recent_reflections: Rolling window of recent serialised reflections.
        agent_output: Set by :class:`~ace.pipeline.steps.AgentStep`.
        environment_result: Set by :class:`~ace.pipeline.steps.EvaluateStep`.
        reflection: Set by :class:`~ace.pipeline.steps.ReflectStep`.
        skill_manager_output: Set by :class:`~ace.pipeline.steps.UpdateStep`.
        metadata: Free-form bag for custom steps or hooks.
    """

    sample: Sample
    skillbook: Skillbook
    environment: TaskEnvironment
    epoch: int
    total_epochs: int
    step_index: int
    total_steps: int
    recent_reflections: List[str] = field(default_factory=list)
    agent_output: Optional[AgentOutput] = None
    environment_result: Optional[EnvironmentResult] = None
    reflection: Optional[ReflectorOutput] = None
    skill_manager_output: Optional[SkillManagerOutput] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step protocol — structural interface for pipeline steps
# ---------------------------------------------------------------------------


@runtime_checkable
class Step(Protocol):
    """Structural protocol for a single pipeline step.

    A step receives the current :class:`StepContext`, performs its work,
    and returns the (possibly mutated) context.  Steps are composable:
    they can be inserted, replaced, or reordered in a pipeline's
    :attr:`ACEPipeline.steps` list.

    Example::

        class LoggingStep:
            def __call__(self, ctx: StepContext) -> StepContext:
                print(f"Processing sample {ctx.step_index}/{ctx.total_steps}")
                return ctx

    """

    def __call__(self, ctx: StepContext) -> StepContext: ...


# ---------------------------------------------------------------------------
# ACEPipeline ABC — shared orchestration and extension point
# ---------------------------------------------------------------------------


class ACEPipeline(ABC):
    """Abstract base class for ACE orchestration pipelines.

    Subclasses implement :meth:`run` to define the iteration strategy
    (e.g. multi-epoch offline training vs. online streaming), and override
    :meth:`_default_steps` to define the default step chain.

    The step chain is fully composable: pass a custom ``steps`` list to
    replace the default Agent → Evaluate → Reflect → Update sequence with
    anything that satisfies the :class:`Step` protocol.

    Args:
        skillbook: Initial skillbook (creates an empty one when *None*).
        agent: Agent instance for producing answers.
        reflector: Reflector instance for analysing outcomes.
        skill_manager: SkillManager instance for updating the skillbook.
        max_refinement_rounds: Max reflection refinement rounds (default: 1).
        reflection_window: Rolling window of recent reflections (default: 3).
        enable_observability: Enable Opik observability if available (default: True).
        async_learning: Enable async Reflector parallelism (default: False).
        max_reflector_workers: Max concurrent Reflector threads (default: 3).
        on_learning_error: Callback on async task error ``(exc, task)``.
        on_learning_complete: Callback on async task complete ``(task, output)``.
        dedup_config: Optional deduplication config for the SkillManager.
        steps: Custom step list — overrides :meth:`_default_steps` when provided.
    """

    def __init__(
        self,
        *,
        skillbook: Optional[Skillbook] = None,
        agent: Agent,
        reflector: Reflector,
        skill_manager: SkillManager,
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
        enable_observability: bool = True,
        async_learning: bool = False,
        max_reflector_workers: int = 3,
        on_learning_error: Optional[Callable[[Exception, Any], None]] = None,
        on_learning_complete: Optional[Callable[[Any, Any], None]] = None,
        dedup_config: Optional["DeduplicationConfig"] = None,
        steps: Optional[List[Step]] = None,
    ) -> None:
        self.skillbook = skillbook or Skillbook()
        self.agent = agent
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window
        self._recent_reflections: List[str] = []

        # Step chain — built after all attrs are set so _default_steps can
        # reference self.agent, self.reflector, etc.
        self.steps: List[Step] = steps if steps is not None else self._default_steps()

        # Async learning
        self._async_learning = async_learning
        self._max_reflector_workers = max_reflector_workers
        self._on_learning_error = on_learning_error
        self._on_learning_complete = on_learning_complete
        self._async_pipeline: Optional["AsyncLearningPipeline"] = None

        # Deduplication
        if dedup_config is not None and skill_manager.dedup_manager is None:
            from ..deduplication import DeduplicationManager

            skill_manager.dedup_manager = DeduplicationManager(dedup_config)
            logger.info(
                f"Deduplication enabled with threshold={dedup_config.similarity_threshold}"
            )

        # Observability
        self.enable_observability = enable_observability
        self.opik_integration: Optional["OpikIntegration"] = None
        if enable_observability:
            try:
                from ..observability import get_integration

                self.opik_integration = get_integration()
            except ImportError:
                self.opik_integration = None
                self.enable_observability = False

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        samples: Any,
        environment: TaskEnvironment,
        **kwargs: Any,
    ) -> List[ACEStepResult]:
        """Run the pipeline over a collection of samples.

        Subclasses define the iteration strategy (epochs, streaming, etc.).
        """
        ...

    def _default_steps(self) -> List[Step]:
        """Return the default step chain for this pipeline.

        Override in subclasses to customise the default composition.
        The base implementation returns an empty list — concrete pipelines
        should always override this.
        """
        return []

    # ------------------------------------------------------------------
    # Core sample processing
    # ------------------------------------------------------------------

    def _process_sample(
        self,
        sample: Sample,
        environment: TaskEnvironment,
        *,
        epoch: int,
        total_epochs: int,
        step_index: int,
        total_steps: int,
    ) -> ACEStepResult:
        """Run the full step chain for a single sample (synchronous)."""
        ctx = StepContext(
            sample=sample,
            skillbook=self.skillbook,
            environment=environment,
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=step_index,
            total_steps=total_steps,
            recent_reflections=list(self._recent_reflections),
        )

        for step in self.steps:
            ctx = step(ctx)

        # Sync rolling reflection window back to pipeline state.
        # ReflectStep may have appended to ctx.recent_reflections.
        self._recent_reflections = ctx.recent_reflections

        if (
            self.enable_observability
            and ctx.agent_output
            and ctx.environment_result
            and ctx.reflection
            and ctx.skill_manager_output
        ):
            self._track_observability_data(
                sample,
                ctx.agent_output,
                ctx.environment_result,
                ctx.reflection,
                ctx.skill_manager_output,
                epoch,
                step_index,
            )

        return ACEStepResult(
            sample=ctx.sample,
            agent_output=ctx.agent_output,
            environment_result=ctx.environment_result,
            reflection=ctx.reflection,
            skill_manager_output=ctx.skill_manager_output,
            skillbook_snapshot=self.skillbook.as_prompt(),
            epoch=epoch,
            step=step_index,
        )

    def _process_sample_async(
        self,
        sample: Sample,
        environment: TaskEnvironment,
        *,
        epoch: int,
        total_epochs: int,
        step_index: int,
        total_steps: int,
    ) -> ACEStepResult:
        """Run sync steps then defer Reflect+Update to the async pipeline.

        The first two steps in :attr:`steps` (Agent, Evaluate) run in the
        calling thread.  The remainder (Reflect, Update) are submitted to
        :class:`~ace.async_learning.AsyncLearningPipeline` and processed
        in the background.
        """
        from ..async_learning import LearningTask

        ctx = StepContext(
            sample=sample,
            skillbook=self.skillbook,
            environment=environment,
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=step_index,
            total_steps=total_steps,
            recent_reflections=list(self._recent_reflections),
        )

        # Run the first two steps synchronously (Agent, Evaluate)
        for step in self.steps[:2]:
            ctx = step(ctx)

        # Submit the rest (Reflect, Update) to the background pipeline
        if self._async_pipeline and ctx.agent_output and ctx.environment_result:
            task = LearningTask(
                sample=sample,
                agent_output=ctx.agent_output,
                environment_result=ctx.environment_result,
                epoch=epoch,
                step_index=step_index,
                total_epochs=total_epochs,
                total_steps=total_steps,
            )
            self._async_pipeline.submit(task)

        return ACEStepResult(
            sample=ctx.sample,
            agent_output=ctx.agent_output,
            environment_result=ctx.environment_result,
            reflection=None,
            skill_manager_output=None,
            skillbook_snapshot=self.skillbook.as_prompt(),
            epoch=epoch,
            step=step_index,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _track_observability_data(
        self,
        sample: Sample,
        agent_output: AgentOutput,
        environment_result: EnvironmentResult,
        reflection: ReflectorOutput,
        skill_manager_output: SkillManagerOutput,
        epoch: int,
        step: int,
    ) -> None:
        """Track data for observability analysis."""
        if not self.enable_observability or not self.opik_integration:
            return

        sample_id = sample.metadata.get("sample_id", f"sample_{step}")

        performance_score = 0.0
        if environment_result.metrics:
            score_metrics = []
            for key, value in environment_result.metrics.items():
                if key in [
                    "correct",
                    "efficient",
                    "success",
                    "accuracy",
                    "score",
                    "syntax_valid",
                    "contains_required",
                ]:
                    if isinstance(value, (int, float, bool)):
                        score_metrics.append(float(value))
            if score_metrics:
                performance_score = sum(score_metrics) / len(score_metrics)

        try:
            self.opik_integration.log_adaptation_metrics(
                epoch=epoch,
                step=step,
                performance_score=performance_score,
                skill_count=len(self.skillbook.skills()),
                successful_predictions=1 if performance_score > 0.5 else 0,
                total_predictions=1,
                metadata={
                    "sample_id": sample_id,
                    "question": (
                        sample.question[:100] + "..."
                        if len(sample.question) > 100
                        else sample.question
                    ),
                    "skill_ids_used": agent_output.skill_ids,
                    "environment_metrics": environment_result.metrics,
                },
            )
        except Exception as e:
            logger.debug(f"Opik observability error (non-critical): {e}")

    def get_observability_data(self) -> Dict[str, Any]:
        """Return observability data (if Opik integration is active)."""
        if not self.enable_observability or not self.opik_integration:
            return {}
        return {
            "observability_enabled": True,
            "opik_available": self.opik_integration.is_available(),
            "skillbook_stats": self.skillbook.stats(),
        }

    # ------------------------------------------------------------------
    # Async learning control
    # ------------------------------------------------------------------

    def _setup_async_pipeline(self) -> None:
        """Initialise the async learning pipeline (idempotent)."""
        if self._async_pipeline is not None:
            return

        from ..async_learning import AsyncLearningPipeline

        self._async_pipeline = AsyncLearningPipeline(
            skillbook=self.skillbook,
            reflector=self.reflector,
            skill_manager=self.skill_manager,
            max_reflector_workers=self._max_reflector_workers,
            max_refinement_rounds=self.max_refinement_rounds,
            on_error=self._on_learning_error,
            on_complete=self._on_learning_complete,
        )

    def start_async_learning(self) -> None:
        """Start the async learning pipeline."""
        if not self._async_learning:
            return
        self._setup_async_pipeline()
        if self._async_pipeline:
            self._async_pipeline.start()

    def stop_async_learning(self, wait: bool = True, timeout: float = 30.0) -> int:
        """Stop the async learning pipeline.

        Returns:
            Number of tasks still remaining in queues.
        """
        if self._async_pipeline is None:
            return 0
        return self._async_pipeline.stop(wait=wait, timeout=timeout)

    def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """Block until all pending learning tasks complete.

        Returns:
            *True* if all tasks completed, *False* on timeout.
        """
        if self._async_pipeline is None:
            return True
        return self._async_pipeline.wait_for_completion(timeout=timeout)

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """Async learning statistics (tasks submitted, completed, failed, etc.)."""
        if self._async_pipeline is None:
            return {
                "tasks_submitted": 0,
                "reflections_completed": 0,
                "skill_updates_completed": 0,
                "tasks_failed": 0,
                "skill_manager_queue_size": 0,
                "is_running": False,
            }
        return self._async_pipeline.stats

    @property
    def is_async_learning(self) -> bool:
        """*True* when async learning mode is enabled."""
        return self._async_learning


# Backward-compatible alias
ACEBase = ACEPipeline
