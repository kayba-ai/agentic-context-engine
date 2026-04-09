"""RRStep — Recursive Reflector pipeline step.

Thin step definition that satisfies ``StepProtocol`` and ``ReflectorLike``.
All implementation logic lives in :mod:`ace.implementations.rr`.
"""

from __future__ import annotations

from typing import Any, Optional

from ace.core.context import ACEStepContext
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.implementations.rr.config import RecursiveConfig as RRConfig
from ace.implementations.rr.runner import RecursiveReflector
from ace.implementations.rr.tools import RRDeps
from ace.core.sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox

from pydantic_ai.settings import ModelSettings

from ace.implementations.rr.prompts import REFLECTOR_RECURSIVE_PROMPT


class RRStep:
    """Recursive Reflector as a pipeline step.

    Satisfies **StepProtocol** (``requires``/``provides``) and
    **ReflectorLike** (``reflect`` method). Delegates all analysis
    to :class:`~ace.implementations.rr.runner.RecursiveReflector`.
    """

    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    def __init__(
        self,
        model: str,
        config: Optional[RRConfig] = None,
        prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self._impl = RecursiveReflector(
            model,
            config=config,
            prompt_template=prompt_template,
            model_settings=model_settings,
        )
        # Expose config for tests and introspection
        self.config = self._impl.config

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        """Run the Recursive Reflector and attach reflection(s) to context."""
        return self._impl(ctx)

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any = None,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """ReflectorLike protocol — delegates to implementation."""
        return self._impl.reflect(
            question=question,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )


__all__ = [
    "RRConfig",
    "RRDeps",
    "RRStep",
    "ExecutionResult",
    "ExecutionTimeoutError",
    "TraceSandbox",
]
