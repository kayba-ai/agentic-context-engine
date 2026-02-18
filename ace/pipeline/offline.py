"""Offline (multi-epoch) ACE pipeline."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence

from ..environments import ACEStepResult, Sample, TaskEnvironment
from ..roles import Agent, Reflector, SkillManager
from ..skillbook import Skillbook
from .base import ACEPipeline, Step
from .steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep

logger = logging.getLogger(__name__)


class OfflineACE(ACEPipeline):
    """Offline ACE pipeline — multi-epoch training over a fixed sample set.

    Processes a fixed training set multiple times (epochs), updating the
    skillbook after each sample.  Useful for building a robust initial
    skillbook before deployment.

    The default step chain (Agent → Evaluate → Reflect → Update) can be
    replaced by passing a custom ``steps`` list.

    Example::

        ace = OfflineACE(agent=agent, reflector=reflector, skill_manager=sm)
        results = ace.run(samples, environment, epochs=3)
        print(ace.skillbook.as_prompt())

    With checkpointing::

        results = ace.run(
            samples, environment, epochs=3,
            checkpoint_interval=10,
            checkpoint_dir="./checkpoints",
        )
    """

    def _default_steps(self) -> List[Step]:
        return [
            AgentStep(self.agent),
            EvaluateStep(),
            ReflectStep(
                self.reflector,
                self.max_refinement_rounds,
                self.reflection_window,
            ),
            UpdateStep(self.skill_manager),
        ]

    def run(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        wait_for_learning: bool = True,
        **kwargs: Any,
    ) -> List[ACEStepResult]:
        """Run offline training over a fixed set of samples.

        Args:
            samples: Training samples to process.
            environment: Environment for evaluating agent outputs.
            epochs: Number of times to iterate over samples (default: 1).
            checkpoint_interval: Save skillbook every N samples (optional).
            checkpoint_dir: Directory for checkpoints (required when
                *checkpoint_interval* is set).
            wait_for_learning: In async mode, block until all background
                tasks complete before returning (default: True).

        Returns:
            List of :class:`~ace.environments.ACEStepResult` for each
            processed sample across all epochs.

        Note:
            Failed samples are logged and skipped; training continues.
            In async mode with ``wait_for_learning=False``, use
            :meth:`wait_for_learning` to block later.
        """
        from pathlib import Path

        if checkpoint_interval is not None and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be provided when checkpoint_interval is set"
            )

        if self._async_learning:
            self.start_async_learning()

        results: List[ACEStepResult] = []
        failed_samples: List[tuple] = []
        total_steps = len(samples)

        try:
            for epoch_idx in range(1, epochs + 1):
                for step_idx, sample in enumerate(samples, start=1):
                    try:
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

                        if (
                            checkpoint_interval
                            and checkpoint_dir
                            and len(results) % checkpoint_interval == 0
                        ):
                            checkpoint_path = Path(checkpoint_dir)
                            numbered = (
                                checkpoint_path / f"ace_checkpoint_{len(results)}.json"
                            )
                            latest = checkpoint_path / "ace_latest.json"
                            self.skillbook.save_to_file(str(numbered))
                            self.skillbook.save_to_file(str(latest))
                            logger.info(
                                f"Checkpoint saved: {len(results)} samples → {numbered.name}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to process sample {step_idx}/{total_steps} "
                            f"in epoch {epoch_idx}/{epochs}: "
                            f"{type(e).__name__}: {str(e)[:200]}"
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

        finally:
            if self._async_learning and wait_for_learning:
                self.wait_for_learning()
                self.stop_async_learning(wait=True)
