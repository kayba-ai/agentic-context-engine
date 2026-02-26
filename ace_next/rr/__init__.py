"""Recursive Reflector as a pipeline step (SubRunner pattern).

Public API::

    from ace_next.rr import RRStep, RRConfig

    rr = RRStep(llm, config=RRConfig(max_iterations=10))
    pipe = Pipeline([..., rr, ...])
"""

from ace.reflector.config import RecursiveConfig as RRConfig

from .context import RRIterationContext
from .runner import RRStep
from .steps import CheckResultStep, ExtractCodeStep, LLMCallStep, SandboxExecStep

__all__ = [
    "RRConfig",
    "RRIterationContext",
    "RRStep",
    "CheckResultStep",
    "ExtractCodeStep",
    "LLMCallStep",
    "SandboxExecStep",
]
