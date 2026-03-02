"""Recursive Reflector as a pipeline step (SubRunner pattern).

Public API::

    from ace_next.rr import RRStep, RRConfig
    from ace_next.rr import ClaudeRRStep, ClaudeRRConfig  # Claude Agent SDK variant

    rr = RRStep(llm, config=RRConfig(max_iterations=10))
    pipe = Pipeline([..., rr, ...])
"""

from ace.reflector.config import RecursiveConfig as RRConfig

from .claude_rr import ClaudeRRConfig, ClaudeRRStep
from .context import RRIterationContext
from .runner import RRStep
from .steps import CheckResultStep, ExtractCodeStep, LLMCallStep, SandboxExecStep

__all__ = [
    "ClaudeRRConfig",
    "ClaudeRRStep",
    "RRConfig",
    "RRIterationContext",
    "RRStep",
    "CheckResultStep",
    "ExtractCodeStep",
    "LLMCallStep",
    "SandboxExecStep",
]
