"""Online (streaming) ACE pipeline."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Optional

from ..environments import ACEStepResult, Sample, TaskEnvironment
from ..roles import Agent, Reflector, SkillManager
from ..skillbook import Skillbook
from .base import ACEPipeline, Step
from .steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep

logger = logging.getLogger(__name__)


class OnlineACE(ACEPipeline):
    """Online ACE pipeline — continuous learning from a stream of samples.

    Processes samples sequentially as they arrive, updating the skillbook
    after each one.  Ideal for production deployment where the agent should
    improve continuously without a separate training phase.

    The default step chain (Agent → Evaluate → Reflect → Update) can be
    replaced by passing a custom ``steps`` list.

    Example::

        ace = OnlineACE(agent=agent, reflector=reflector, skill_manager=sm)
        results = ace.run(sample_stream(), environment)
        print(f"Skills learned: {len(ace.skillbook.skills())}")

    Online vs Offline:
        - **Online**: each sample processed once, skillbook adapts immediately.
        - **Offline**: fixed set processed over multiple epochs for thorough learning.
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
        samples: Iterable[Sample],
        environment: TaskEnvironment,
        wait_for_learning: bool = True,
        **kwargs: Any,
    ) -> List[ACEStepResult]:
        """Run the online pipeline over a (potentially infinite) stream of samples.

        Args:
            samples: Iterable of samples — can be an infinite generator.
            environment: Environment for evaluating agent outputs.
            wait_for_learning: In async mode, block until all background
                tasks complete before returning (default: True).

        Returns:
            List of :class:`~ace.environments.ACEStepResult` for each
            processed sample.
        """
        if self._async_learning:
            self.start_async_learning()

        try:
            results: List[ACEStepResult] = []
            for step_idx, sample in enumerate(samples, start=1):
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
            if self._async_learning and wait_for_learning:
                self.wait_for_learning()
                self.stop_async_learning(wait=True)
