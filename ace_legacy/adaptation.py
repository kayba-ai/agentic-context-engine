"""ACE adaptation loop — processes samples through the Agent→Reflector→SkillManager pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
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


class ACE:
    """
    Core ACE adaptation loop.

    Processes samples through the pipeline:
        Agent → Environment → Reflector → SkillManager → Skillbook

    Works for both offline (ReplayAgent + traces) and online (Agent + live generation).
    The distinction is just which agent you pass in.
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
    ) -> None:
        self.skillbook = skillbook or Skillbook()
        self.agent = agent
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window
        self._recent_reflections: List[str] = []

    def run(
        self,
        samples: Iterable[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> List[ACEStepResult]:
        """
        Run the ACE loop over samples.

        Args:
            samples: Samples to process (list for multi-epoch, any iterable for streaming)
            environment: Environment for evaluating agent outputs
            epochs: Number of passes over samples (default: 1)
            checkpoint_interval: Save skillbook every N successful samples (optional)
            checkpoint_dir: Directory to save checkpoints (required if checkpoint_interval set)

        Returns:
            List of ACEStepResult for each processed sample
        """
        if checkpoint_interval is not None and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be provided when checkpoint_interval is set"
            )

        # Materialise once so we can iterate multiple epochs
        sample_list = list(samples)
        total_steps = len(sample_list)

        results: List[ACEStepResult] = []
        failed_samples: List[tuple] = []

        for epoch_idx in range(1, epochs + 1):
            for step_idx, sample in enumerate(sample_list, start=1):
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

                    if (
                        checkpoint_interval
                        and checkpoint_dir
                        and len(results) % checkpoint_interval == 0
                    ):
                        cp = Path(checkpoint_dir)
                        cp.mkdir(parents=True, exist_ok=True)
                        self.skillbook.save_to_file(str(cp / f"ace_checkpoint_{len(results)}.json"))
                        self.skillbook.save_to_file(str(cp / "ace_latest.json"))
                        logger.info(f"Checkpoint saved: {len(results)} samples")

                except Exception as e:
                    logger.warning(
                        f"Failed sample {step_idx}/{total_steps} "
                        f"epoch {epoch_idx}/{epochs}: {type(e).__name__}: {str(e)[:200]}"
                    )
                    failed_samples.append((epoch_idx, step_idx, str(e)[:100]))
                    continue

        if failed_samples:
            logger.info(
                f"Completed with {len(failed_samples)} failed samples "
                f"out of {total_steps * epochs} total"
            )

        return results

    # ── internals ───────────────────────────────────────────────────────

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


# Backwards-compat aliases
OfflineACE = ACE
OnlineACE = ACE
ACEBase = ACE
