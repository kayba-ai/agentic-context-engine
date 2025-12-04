"""Adaptation loops for offline and online ACE training."""

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
    Sequence,
)

if TYPE_CHECKING:
    from .async_learning import AsyncLearningPipeline
    from .deduplication import DeduplicationConfig
    from .observability.opik_integration import OpikIntegration

from .skillbook import Skillbook
from .roles import (
    SkillManager,
    SkillManagerOutput,
    Agent,
    AgentOutput,
    Reflector,
    ReflectorOutput,
)

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Single task instance presented to ACE."""

    question: str
    context: str = ""
    ground_truth: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EnvironmentResult:
    """Feedback returned by the task environment after executing the generator output."""

    feedback: str
    ground_truth: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


class TaskEnvironment(ABC):
    """
    Abstract interface for evaluating agent outputs.

    Implement this class to define how your specific task evaluates
    the Agent's answers. The environment provides feedback that
    helps ACE learn what works and what doesn't.

    Example Implementation:
        >>> class MathEnvironment(TaskEnvironment):
        ...     def evaluate(self, sample, agent_output):
        ...         # Parse the answer
        ...         predicted = extract_number(agent_output.final_answer)
        ...         correct = str(predicted) == sample.ground_truth
        ...
        ...         # Provide feedback
        ...         if correct:
        ...             feedback = "Correct!"
        ...         else:
        ...             feedback = f"Incorrect. Expected {sample.ground_truth}"
        ...
        ...         return EnvironmentResult(
        ...             feedback=feedback,
        ...             ground_truth=sample.ground_truth,
        ...             metrics={'accuracy': 1.0 if correct else 0.0}
        ...         )
    """

    @abstractmethod
    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        """
        Evaluate the agent's output for a given sample.

        Args:
            sample: The input sample with question and context
            agent_output: The Agent's produced answer

        Returns:
            EnvironmentResult with feedback and optional ground truth

        The feedback should be informative enough for the Reflector
        to understand what went right or wrong.
        """


class SimpleEnvironment(TaskEnvironment):
    """
    Simple built-in environment for quick testing and demos.

    Checks if the ground truth appears in the answer (case-insensitive).
    Perfect for getting started without creating a custom environment.

    Example:
        >>> from ace import SimpleEnvironment, Sample
        >>> env = SimpleEnvironment()
        >>> sample = Sample(question="What is 2+2?", ground_truth="4")
        >>> result = env.evaluate(sample, agent_output)
    """

    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        """Check if ground truth appears in the answer."""
        if not sample.ground_truth:
            return EnvironmentResult(
                feedback="No ground truth provided",
                ground_truth=None,
                metrics={"correct": 0.0},
            )

        answer = agent_output.final_answer.lower()
        truth = sample.ground_truth.lower()
        is_correct = truth in answer

        return EnvironmentResult(
            feedback=(
                "Correct!"
                if is_correct
                else f"Incorrect. Expected: {sample.ground_truth}"
            ),
            ground_truth=sample.ground_truth,
            metrics={"correct": 1.0 if is_correct else 0.0},
        )


@dataclass
class ACEStepResult:
    """Result from processing a single sample through the ACE pipeline.

    In sync mode, all fields are populated immediately.
    In async mode, reflection and skill_manager_output may be None initially
    (they are processed in background).
    """

    sample: Sample
    agent_output: AgentOutput
    environment_result: EnvironmentResult
    reflection: Optional[ReflectorOutput]  # None in async mode until processed
    skill_manager_output: Optional[
        SkillManagerOutput
    ]  # None in async mode until processed
    skillbook_snapshot: str

    # Observability metadata
    epoch: int = 0
    step: int = 0
    performance_score: float = 0.0


class ACEBase:
    """Shared orchestration logic for offline and online ACE adaptation."""

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
        # Async learning parameters
        async_learning: bool = False,
        max_reflector_workers: int = 3,
        on_learning_error: Optional[Callable[[Exception, Any], None]] = None,
        on_learning_complete: Optional[Callable[[Any, Any], None]] = None,
        # Deduplication
        dedup_config: Optional["DeduplicationConfig"] = None,
    ) -> None:
        self.skillbook = skillbook or Skillbook()
        self.agent = agent
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window
        self._recent_reflections: List[str] = []

        # Async learning configuration
        self._async_learning = async_learning
        self._max_reflector_workers = max_reflector_workers
        self._on_learning_error = on_learning_error
        self._on_learning_complete = on_learning_complete
        self._async_pipeline: Optional[AsyncLearningPipeline] = None

        # Set up deduplication if config provided and skill_manager doesn't have one
        if dedup_config is not None and skill_manager.dedup_manager is None:
            from .deduplication import DeduplicationManager

            skill_manager.dedup_manager = DeduplicationManager(dedup_config)
            logger.info(
                f"Deduplication enabled with threshold={dedup_config.similarity_threshold}"
            )

        # Observability integration
        self.enable_observability = enable_observability
        self.opik_integration: Optional[OpikIntegration] = None
        if enable_observability:
            try:
                from .observability import get_integration

                self.opik_integration = get_integration()
            except ImportError:
                self.opik_integration = None
                self.enable_observability = False

    # ------------------------------------------------------------------ #
    # Observability tracking methods
    # ------------------------------------------------------------------ #
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

        # Calculate performance score from success/quality metrics only
        performance_score = 0.0
        if environment_result.metrics:
            # Only use boolean/probability metrics that represent success/quality (0-1 range)
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

        # Track adaptation metrics with Opik
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
            # Log observability errors in debug mode but don't interrupt main flow
            logger.debug(f"Opik observability error (non-critical): {e}")

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data (if available through Opik integration)."""
        if not self.enable_observability or not self.opik_integration:
            return {}

        return {
            "observability_enabled": True,
            "opik_available": self.opik_integration.is_available(),
            "skillbook_stats": self.skillbook.stats(),
        }

    # ------------------------------------------------------------------ #
    # Async learning control methods
    # ------------------------------------------------------------------ #
    def _setup_async_pipeline(self) -> None:
        """Initialize the async learning pipeline."""
        if self._async_pipeline is not None:
            return

        from .async_learning import AsyncLearningPipeline

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
        """Start the async learning pipeline.

        Call this before processing samples in async mode.
        """
        if not self._async_learning:
            return

        self._setup_async_pipeline()
        if self._async_pipeline:
            self._async_pipeline.start()

    def stop_async_learning(self, wait: bool = True, timeout: float = 30.0) -> int:
        """Stop the async learning pipeline.

        Args:
            wait: If True, wait for pending tasks to complete
            timeout: Max seconds to wait for completion

        Returns:
            Number of tasks remaining in queues
        """
        if self._async_pipeline is None:
            return 0

        remaining = self._async_pipeline.stop(wait=wait, timeout=timeout)
        return remaining

    def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending learning tasks to complete.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if all tasks completed, False if timeout
        """
        if self._async_pipeline is None:
            return True

        return self._async_pipeline.wait_for_completion(timeout=timeout)

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """Get async learning statistics.

        Returns:
            Dict with tasks_submitted, reflections_completed, skill_updates_completed,
            tasks_failed, skill_manager_queue_size, is_running
        """
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
        """Check if async learning mode is enabled."""
        return self._async_learning

    # ------------------------------------------------------------------ #
    def _reflection_context(self) -> str:
        return "\n---\n".join(self._recent_reflections)

    def _update_recent_reflections(self, reflection: ReflectorOutput) -> None:
        serialized = json.dumps(reflection.raw, ensure_ascii=False)
        self._recent_reflections.append(serialized)
        if len(self._recent_reflections) > self.reflection_window:
            self._recent_reflections = self._recent_reflections[
                -self.reflection_window :
            ]

    def _apply_skill_tags(self, reflection: ReflectorOutput) -> None:
        for tag in reflection.skill_tags:
            try:
                self.skillbook.tag_skill(tag.id, tag.tag)
            except ValueError:
                continue

    def _question_context(
        self, sample: Sample, environment_result: EnvironmentResult
    ) -> str:
        parts = [
            f"question: {sample.question}",
            f"context: {sample.context}",
            f"metadata: {json.dumps(sample.metadata)}",
            f"feedback: {environment_result.feedback}",
            f"ground_truth: {environment_result.ground_truth}",
        ]
        return "\n".join(parts)

    def _progress_string(
        self, epoch: int, total_epochs: int, step: int, total_steps: int
    ) -> str:
        return f"epoch {epoch}/{total_epochs} · sample {step}/{total_steps}"

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
        agent_output = self.agent.generate(
            question=sample.question,
            context=sample.context,
            skillbook=self.skillbook,
            reflection=self._reflection_context(),
            sample=sample,  # Pass sample for ReplayAgent support
        )
        env_result = environment.evaluate(sample, agent_output)
        reflection = self.reflector.reflect(
            question=sample.question,
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
            max_refinement_rounds=self.max_refinement_rounds,
        )
        self._apply_skill_tags(reflection)
        self._update_recent_reflections(reflection)
        skill_manager_output = self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=self._question_context(sample, env_result),
            progress=self._progress_string(
                epoch, total_epochs, step_index, total_steps
            ),
        )

        # Track observability data if enabled
        if self.enable_observability:
            self._track_observability_data(
                sample,
                agent_output,
                env_result,
                reflection,
                skill_manager_output,
                epoch,
                step_index,
            )

        self.skillbook.apply_update(skill_manager_output.update)

        return ACEStepResult(
            sample=sample,
            agent_output=agent_output,
            environment_result=env_result,
            reflection=reflection,
            skill_manager_output=skill_manager_output,
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
        """Process sample with async learning - Agent returns immediately.

        Learning (Reflector -> SkillManager -> Skillbook) happens in background.
        """
        from .async_learning import LearningTask

        # Generate (uses current skillbook - eventual consistency)
        agent_output = self.agent.generate(
            question=sample.question,
            context=sample.context,
            skillbook=self.skillbook,
            reflection=self._reflection_context(),
            sample=sample,
        )

        # Evaluate (sync - usually fast)
        env_result = environment.evaluate(sample, agent_output)

        # Submit to async pipeline (Reflector runs in thread pool)
        if self._async_pipeline:
            task = LearningTask(
                sample=sample,
                agent_output=agent_output,
                environment_result=env_result,
                epoch=epoch,
                step_index=step_index,
                total_epochs=total_epochs,
                total_steps=total_steps,
            )
            self._async_pipeline.submit(task)

        # Return immediately with partial result
        return ACEStepResult(
            sample=sample,
            agent_output=agent_output,
            environment_result=env_result,
            reflection=None,  # Processing in background
            skill_manager_output=None,  # Processing in background
            skillbook_snapshot=self.skillbook.as_prompt(),
            epoch=epoch,
            step=step_index,
        )


class OfflineACE(ACEBase):
    """
    Orchestrates offline ACE adaptation over multiple training epochs.

    The OfflineACE processes a fixed training set multiple times,
    allowing the skillbook to evolve and improve through repeated exposure
    to the same examples. This is useful for building a robust initial
    skillbook before deployment.

    Args:
        skillbook: Initial skillbook (creates empty one if None)
        agent: Agent instance for producing answers
        reflector: Reflector instance for analyzing outcomes
        skill_manager: SkillManager instance for updating skillbook
        max_refinement_rounds: Max reflection refinement attempts (default: 1)
        reflection_window: Number of recent reflections to maintain (default: 3)
        dedup_config: Optional DeduplicationConfig for skill deduplication

    Example:
        >>> from ace import OfflineACE, Agent, Reflector, SkillManager
        >>> from ace.llm_providers import LiteLLMClient
        >>>
        >>> # Initialize components with same LLM
        >>> client = LiteLLMClient(model="gpt-4")
        >>> agent = Agent(client)
        >>> reflector = Reflector(client)
        >>> skill_manager = SkillManager(client)
        >>>
        >>> # Create ACE
        >>> ace = OfflineACE(
        ...     agent=agent,
        ...     reflector=reflector,
        ...     skill_manager=skill_manager
        ... )
        >>>
        >>> # Prepare training samples
        >>> samples = [
        ...     Sample(question="What is 2+2?", ground_truth="4"),
        ...     Sample(question="What is 5*3?", ground_truth="15")
        ... ]
        >>>
        >>> # Run adaptation for 3 epochs
        >>> results = ace.run(samples, environment, epochs=3)
        >>>
        >>> # Access evolved skillbook
        >>> print(ace.skillbook.as_prompt())

    With Deduplication:
        >>> from ace.deduplication import DeduplicationConfig
        >>>
        >>> # Enable skill deduplication with custom threshold
        >>> dedup_config = DeduplicationConfig(
        ...     similarity_threshold=0.85,
        ...     embedding_model="text-embedding-3-small"
        ... )
        >>> ace = OfflineACE(
        ...     agent=agent,
        ...     reflector=reflector,
        ...     skill_manager=skill_manager,
        ...     dedup_config=dedup_config
        ... )
        >>> # Similar skills will now be detected and reported to SkillManager

    The ACE will:
        1. Process each sample through Agent → Environment → Reflector → SkillManager
        2. Update the skillbook after each sample
        3. Repeat for the specified number of epochs
        4. Return detailed results for analysis
    """

    def run(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        wait_for_learning: bool = True,
    ) -> List[ACEStepResult]:
        """
        Run offline adaptation over training samples.

        Args:
            samples: Training samples to process
            environment: Environment for evaluating agent outputs
            epochs: Number of times to iterate over samples (default: 1)
            checkpoint_interval: Save skillbook every N successful samples (optional)
            checkpoint_dir: Directory to save checkpoints (required if checkpoint_interval set)
            wait_for_learning: If async_learning=True, wait for all learning tasks
                to complete before returning (default: True)

        Returns:
            List of ACEStepResult for each processed sample

        Note:
            The skillbook is updated in-place during adaptation.
            Access the evolved skillbook via ace.skillbook after running.
            Failed samples are skipped and logged, training continues.
            In async mode with wait_for_learning=False, learning continues in
            background. Use wait_for_learning() to block when needed.
        """
        from pathlib import Path

        results: List[ACEStepResult] = []
        failed_samples: List[tuple] = (
            []
        )  # Track (epoch, step_idx, error) for failed samples
        total_steps = len(samples)

        # Validate checkpoint parameters
        if checkpoint_interval is not None and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be provided when checkpoint_interval is set"
            )

        # Start async pipeline if enabled
        if self._async_learning:
            self.start_async_learning()

        try:
            for epoch_idx in range(1, epochs + 1):
                for step_idx, sample in enumerate(samples, start=1):
                    try:
                        # Use async or sync processing based on mode
                        if self._async_learning:
                            result = self._process_sample_async(
                                sample,
                                environment,
                                epoch=epoch_idx,
                                total_epochs=epochs,
                                step_index=step_idx,
                                total_steps=total_steps,
                            )
                        else:
                            result = self._process_sample(
                                sample,
                                environment,
                                epoch=epoch_idx,
                                total_epochs=epochs,
                                step_index=step_idx,
                                total_steps=total_steps,
                            )
                        results.append(result)

                        # Save checkpoint if interval reached
                        if (
                            checkpoint_interval
                            and checkpoint_dir
                            and len(results) % checkpoint_interval == 0
                        ):
                            checkpoint_path = Path(checkpoint_dir)
                            numbered_checkpoint = (
                                checkpoint_path / f"ace_checkpoint_{len(results)}.json"
                            )
                            latest_checkpoint = checkpoint_path / "ace_latest.json"

                            self.skillbook.save_to_file(str(numbered_checkpoint))
                            self.skillbook.save_to_file(str(latest_checkpoint))
                            logger.info(
                                f"Checkpoint saved: {len(results)} samples → {numbered_checkpoint.name}"
                            )

                    except Exception as e:
                        # Log error and continue to next sample
                        logger.warning(
                            f"Failed to process sample {step_idx}/{total_steps} "
                            f"in epoch {epoch_idx}/{epochs}: {type(e).__name__}: {str(e)[:200]}"
                        )
                        failed_samples.append((epoch_idx, step_idx, str(e)[:100]))
                        continue

            # Report failure summary if any samples failed
            if failed_samples:
                logger.info(
                    f"Training completed with {len(failed_samples)} failed samples "
                    f"out of {len(samples) * epochs} total attempts"
                )
                logger.debug(f"Failed samples: {failed_samples}")

            return results

        finally:
            # Wait for async learning to complete and stop pipeline
            if self._async_learning and wait_for_learning:
                self.wait_for_learning()
                self.stop_async_learning(wait=True)
            # If wait_for_learning=False, leave pipeline running for user to manage


class OnlineACE(ACEBase):
    """
    Orchestrates online ACE adaptation for continuous learning.

    The OnlineACE processes samples sequentially as they arrive,
    updating the skillbook after each one. This enables continuous
    improvement during deployment, adapting to new patterns and
    correcting mistakes in real-time.

    Args:
        skillbook: Initial skillbook (creates empty one if None)
        agent: Agent instance for producing answers
        reflector: Reflector instance for analyzing outcomes
        skill_manager: SkillManager instance for updating skillbook
        max_refinement_rounds: Max reflection refinement attempts (default: 1)
        reflection_window: Number of recent reflections to maintain (default: 3)
        dedup_config: Optional DeduplicationConfig for skill deduplication

    Example:
        >>> from ace import OnlineACE, Agent, Reflector, SkillManager
        >>> from ace.llm_providers import LiteLLMClient
        >>>
        >>> # Initialize with pre-trained skillbook
        >>> skillbook = Skillbook.load_from_file("pretrained_skillbook.json")
        >>>
        >>> client = LiteLLMClient(model="gpt-4")
        >>> ace = OnlineACE(
        ...     skillbook=skillbook,
        ...     agent=Agent(client),
        ...     reflector=Reflector(client),
        ...     skill_manager=SkillManager(client)
        ... )
        >>>
        >>> # Process streaming samples
        >>> def sample_stream():
        ...     while True:
        ...         yield get_next_sample()  # Your sample source
        >>>
        >>> # Run online adaptation
        >>> results = ace.run(sample_stream(), environment)
        >>>
        >>> # Skillbook evolves with each sample
        >>> print(f"Skills: {len(ace.skillbook.skills())}")

    With Deduplication:
        >>> from ace.deduplication import DeduplicationConfig
        >>>
        >>> dedup_config = DeduplicationConfig(similarity_threshold=0.85)
        >>> ace = OnlineACE(
        ...     skillbook=skillbook,
        ...     agent=Agent(client),
        ...     reflector=Reflector(client),
        ...     skill_manager=SkillManager(client),
        ...     dedup_config=dedup_config
        ... )

    Online vs Offline:
        - Online: Processes each sample once, adapts immediately
        - Offline: Processes fixed set multiple times for thorough learning
        - Online is ideal for production deployment with continuous improvement
        - Offline is ideal for initial training before deployment
    """

    def run(
        self,
        samples: Iterable[Sample],
        environment: TaskEnvironment,
        wait_for_learning: bool = True,
    ) -> List[ACEStepResult]:
        """
        Run online adaptation over a stream of samples.

        Args:
            samples: Iterable of samples (can be infinite stream)
            environment: Environment for evaluating agent outputs
            wait_for_learning: If async_learning=True, wait for all learning tasks
                to complete before returning (default: True)

        Returns:
            List of ACEStepResult for each processed sample

        Note:
            - Processes samples sequentially, updating after each one
            - The skillbook evolves continuously during processing
            - Can handle infinite streams for continuous deployment
            - In async mode with wait_for_learning=False, learning continues in
              background. Use wait_for_learning() to block when needed.
        """
        # Start async pipeline if enabled
        if self._async_learning:
            self.start_async_learning()

        try:
            results: List[ACEStepResult] = []
            step_idx = 0
            for step_idx, sample in enumerate(samples, start=1):
                # Use async or sync processing based on mode
                if self._async_learning:
                    result = self._process_sample_async(
                        sample,
                        environment,
                        epoch=1,
                        total_epochs=1,
                        step_index=step_idx,
                        total_steps=step_idx,
                    )
                else:
                    result = self._process_sample(
                        sample,
                        environment,
                        epoch=1,
                        total_epochs=1,
                        step_index=step_idx,
                        total_steps=step_idx,
                    )
                results.append(result)
            return results

        finally:
            # Wait for async learning to complete and stop pipeline
            if self._async_learning and wait_for_learning:
                self.wait_for_learning()
                self.stop_async_learning(wait=True)
            # If wait_for_learning=False, leave pipeline running for user to manage
