"""Adaptation loops for offline and online ACE training."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .skillbook import Skillbook
from .roles import (
    SkillManager,
    SkillManagerOutput,
    Agent,
    AgentOutput,
    Reflector,
    ReflectorOutput,
)
from .environments import (
    Sample,
    EnvironmentResult,
    TaskEnvironment,
    ACEStepResult,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.skillbook = skillbook or Skillbook()
        self.agent = agent
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window
        self._recent_reflections: List[str] = []

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
            sample=sample,
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


class OfflineACE(ACEBase):
    """
    Orchestrates offline ACE adaptation over multiple training epochs.

    Processes a fixed training set multiple times, allowing the skillbook
    to evolve through repeated exposure to the same examples.
    """

    def run(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> List[ACEStepResult]:
        """
        Run offline adaptation over training samples.

        Args:
            samples: Training samples to process
            environment: Environment for evaluating agent outputs
            epochs: Number of times to iterate over samples (default: 1)
            checkpoint_interval: Save skillbook every N successful samples (optional)
            checkpoint_dir: Directory to save checkpoints (required if checkpoint_interval set)

        Returns:
            List of ACEStepResult for each processed sample
        """
        from pathlib import Path

        results: List[ACEStepResult] = []
        failed_samples: List[tuple] = []
        total_steps = len(samples)

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
                            checkpoint_path / f"ace_checkpoint_{len(results)}.json"
                        )
                        latest_checkpoint = checkpoint_path / "ace_latest.json"

                        self.skillbook.save_to_file(str(numbered_checkpoint))
                        self.skillbook.save_to_file(str(latest_checkpoint))
                        logger.info(
                            f"Checkpoint saved: {len(results)} samples → {numbered_checkpoint.name}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to process sample {step_idx}/{total_steps} "
                        f"in epoch {epoch_idx}/{epochs}: {type(e).__name__}: {str(e)[:200]}"
                    )
                    failed_samples.append((epoch_idx, step_idx, str(e)[:100]))
                    continue

        if failed_samples:
            logger.info(
                f"Training completed with {len(failed_samples)} failed samples "
                f"out of {len(samples) * epochs} total attempts"
            )
            logger.debug(f"Failed samples: {failed_samples}")

        return results


class OnlineACE(ACEBase):
    """
    Orchestrates online ACE adaptation for continuous learning.

    Processes samples sequentially as they arrive, updating the
    skillbook after each one for continuous improvement.
    """

    def run(
        self,
        samples: Iterable[Sample],
        environment: TaskEnvironment,
    ) -> List[ACEStepResult]:
        """
        Run online adaptation over a stream of samples.

        Args:
            samples: Iterable of samples (can be infinite stream)
            environment: Environment for evaluating agent outputs

        Returns:
            List of ACEStepResult for each processed sample
        """
        results: List[ACEStepResult] = []
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
