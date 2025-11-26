"""Adaptation loops for offline and online ACE training."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

if TYPE_CHECKING:
    from .observability.opik_integration import OpikIntegration

from .playbook import Playbook
from .roles import (
    Curator,
    CuratorOutput,
    Generator,
    GeneratorOutput,
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
    Abstract interface for evaluating generator outputs.

    Implement this class to define how your specific task evaluates
    the Generator's answers. The environment provides feedback that
    helps ACE learn what works and what doesn't.

    Example Implementation:
        >>> class MathEnvironment(TaskEnvironment):
        ...     def evaluate(self, sample, generator_output):
        ...         # Parse the answer
        ...         predicted = extract_number(generator_output.final_answer)
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
    def evaluate(
        self, sample: Sample, generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """
        Evaluate the generator's output for a given sample.

        Args:
            sample: The input sample with question and context
            generator_output: The Generator's produced answer

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
        >>> result = env.evaluate(sample, generator_output)
    """

    def evaluate(
        self, sample: Sample, generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """Check if ground truth appears in the answer."""
        if not sample.ground_truth:
            return EnvironmentResult(
                feedback="No ground truth provided",
                ground_truth=None,
                metrics={"correct": 0.0},
            )

        answer = generator_output.final_answer.lower()
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
class AdapterStepResult:
    sample: Sample
    generator_output: GeneratorOutput
    environment_result: EnvironmentResult
    reflection: ReflectorOutput
    curator_output: CuratorOutput
    playbook_snapshot: str

    # Observability metadata
    epoch: int = 0
    step: int = 0
    performance_score: float = 0.0


class AdapterBase:
    """Shared orchestration logic for offline and online ACE adaptation."""

    def __init__(
        self,
        *,
        playbook: Optional[Playbook] = None,
        generator: Generator,
        reflector: Reflector,
        curator: Curator,
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
        enable_observability: bool = True,
    ) -> None:
        self.playbook = playbook or Playbook()
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window
        self._recent_reflections: List[str] = []

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
        generator_output: GeneratorOutput,
        environment_result: EnvironmentResult,
        reflection: ReflectorOutput,
        curator_output: CuratorOutput,
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
                bullet_count=len(self.playbook.bullets()),
                successful_predictions=1 if performance_score > 0.5 else 0,
                total_predictions=1,
                metadata={
                    "sample_id": sample_id,
                    "question": (
                        sample.question[:100] + "..."
                        if len(sample.question) > 100
                        else sample.question
                    ),
                    "bullet_ids_used": generator_output.bullet_ids,
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
            "playbook_stats": self.playbook.stats(),
        }

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

    def _apply_bullet_tags(self, reflection: ReflectorOutput) -> None:
        for tag in reflection.bullet_tags:
            try:
                self.playbook.tag_bullet(tag.id, tag.tag)
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
    ) -> AdapterStepResult:
        generator_output = self.generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=self.playbook,
            reflection=self._reflection_context(),
            sample=sample,  # Pass sample for ReplayGenerator support
        )
        env_result = environment.evaluate(sample, generator_output)
        reflection = self.reflector.reflect(
            question=sample.question,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
            max_refinement_rounds=self.max_refinement_rounds,
        )
        self._apply_bullet_tags(reflection)
        self._update_recent_reflections(reflection)
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=self._question_context(sample, env_result),
            progress=self._progress_string(
                epoch, total_epochs, step_index, total_steps
            ),
        )

        # Track observability data if enabled
        if self.enable_observability:
            self._track_observability_data(
                sample,
                generator_output,
                env_result,
                reflection,
                curator_output,
                epoch,
                step_index,
            )

        self.playbook.apply_delta(curator_output.delta)

        return AdapterStepResult(
            sample=sample,
            generator_output=generator_output,
            environment_result=env_result,
            reflection=reflection,
            curator_output=curator_output,
            playbook_snapshot=self.playbook.as_prompt(),
            epoch=epoch,
            step=step_index,
        )


class OfflineAdapter(AdapterBase):
    """
    Orchestrates offline ACE adaptation over multiple training epochs.

    The OfflineAdapter processes a fixed training set multiple times,
    allowing the playbook to evolve and improve through repeated exposure
    to the same examples. This is useful for building a robust initial
    playbook before deployment.

    Args:
        playbook: Initial playbook (creates empty one if None)
        generator: Generator instance for producing answers
        reflector: Reflector instance for analyzing outcomes
        curator: Curator instance for updating playbook
        max_refinement_rounds: Max reflection refinement attempts (default: 1)
        reflection_window: Number of recent reflections to maintain (default: 3)

    Example:
        >>> from ace import OfflineAdapter, Generator, Reflector, Curator
        >>> from ace.llm_providers import LiteLLMClient
        >>>
        >>> # Initialize components with same LLM
        >>> client = LiteLLMClient(model="gpt-4")
        >>> generator = Generator(client)
        >>> reflector = Reflector(client)
        >>> curator = Curator(client)
        >>>
        >>> # Create adapter
        >>> adapter = OfflineAdapter(
        ...     generator=generator,
        ...     reflector=reflector,
        ...     curator=curator
        ... )
        >>>
        >>> # Prepare training samples
        >>> samples = [
        ...     Sample(question="What is 2+2?", ground_truth="4"),
        ...     Sample(question="What is 5*3?", ground_truth="15")
        ... ]
        >>>
        >>> # Run adaptation for 3 epochs
        >>> results = adapter.run(samples, environment, epochs=3)
        >>>
        >>> # Access evolved playbook
        >>> print(adapter.playbook.as_prompt())

    The adapter will:
        1. Process each sample through Generator → Environment → Reflector → Curator
        2. Update the playbook after each sample
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
    ) -> List[AdapterStepResult]:
        """
        Run offline adaptation over training samples.

        Args:
            samples: Training samples to process
            environment: Environment for evaluating generator outputs
            epochs: Number of times to iterate over samples (default: 1)
            checkpoint_interval: Save playbook every N successful samples (optional)
            checkpoint_dir: Directory to save checkpoints (required if checkpoint_interval set)

        Returns:
            List of AdapterStepResult for each processed sample

        Note:
            The playbook is updated in-place during adaptation.
            Access the evolved playbook via adapter.playbook after running.
            Failed samples are skipped and logged, training continues.
        """
        from pathlib import Path

        results: List[AdapterStepResult] = []
        failed_samples: List[tuple] = (
            []
        )  # Track (epoch, step_idx, error) for failed samples
        total_steps = len(samples)

        # Validate checkpoint parameters
        if checkpoint_interval is not None and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be provided when checkpoint_interval is set"
            )

        for epoch_idx in range(1, epochs + 1):
            for step_idx, sample in enumerate(samples, start=1):
                try:
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
                            checkpoint_path / f"convex_checkpoint_{len(results)}.json"
                        )
                        latest_checkpoint = checkpoint_path / "convex_latest.json"

                        self.playbook.save_to_file(str(numbered_checkpoint))
                        self.playbook.save_to_file(str(latest_checkpoint))
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


class OnlineAdapter(AdapterBase):
    """
    Orchestrates online ACE adaptation for continuous learning.

    The OnlineAdapter processes samples sequentially as they arrive,
    updating the playbook after each one. This enables continuous
    improvement during deployment, adapting to new patterns and
    correcting mistakes in real-time.

    Args:
        playbook: Initial playbook (creates empty one if None)
        generator: Generator instance for producing answers
        reflector: Reflector instance for analyzing outcomes
        curator: Curator instance for updating playbook
        max_refinement_rounds: Max reflection refinement attempts (default: 1)
        reflection_window: Number of recent reflections to maintain (default: 3)

    Example:
        >>> from ace import OnlineAdapter, Generator, Reflector, Curator
        >>> from ace.llm_providers import LiteLLMClient
        >>>
        >>> # Initialize with pre-trained playbook
        >>> playbook = Playbook.load_from_file("pretrained_playbook.json")
        >>>
        >>> client = LiteLLMClient(model="gpt-4")
        >>> adapter = OnlineAdapter(
        ...     playbook=playbook,
        ...     generator=Generator(client),
        ...     reflector=Reflector(client),
        ...     curator=Curator(client)
        ... )
        >>>
        >>> # Process streaming samples
        >>> def sample_stream():
        ...     while True:
        ...         yield get_next_sample()  # Your sample source
        >>>
        >>> # Run online adaptation
        >>> results = adapter.run(sample_stream(), environment)
        >>>
        >>> # Playbook evolves with each sample
        >>> print(f"Bullets: {len(adapter.playbook.bullets)}")

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
    ) -> List[AdapterStepResult]:
        """
        Run online adaptation over a stream of samples.

        Args:
            samples: Iterable of samples (can be infinite stream)
            environment: Environment for evaluating generator outputs

        Returns:
            List of AdapterStepResult for each processed sample

        Note:
            - Processes samples sequentially, updating after each one
            - The playbook evolves continuously during processing
            - Can handle infinite streams for continuous deployment
        """
        results: List[AdapterStepResult] = []
        step_idx = 0
        for step_idx, sample in enumerate(samples, start=1):
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
