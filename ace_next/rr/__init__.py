"""Recursive Reflector as a pipeline step (SubRunner pattern).

Public API::

    from ace_next.rr import RRStep, RRConfig

    rr = RRStep(llm, config=RRConfig(max_iterations=10))
    pipe = Pipeline([..., rr, ...])
"""

from .config import RecursiveConfig as RRConfig
from .context import RRIterationContext
from .runner import RRStep
from .sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox
from .steps import CheckResultStep, ExtractCodeStep, LLMCallStep, SandboxExecStep
from .subagent import (
    CallBudget,
    SubAgentConfig,
    SubAgentLLM,
    create_ask_llm_function,
)
from .trace_context import TraceContext, TraceStep

__all__ = [
    "RRConfig",
    "RRIterationContext",
    "RRStep",
    # Inner pipeline steps
    "CheckResultStep",
    "ExtractCodeStep",
    "LLMCallStep",
    "SandboxExecStep",
    # Sandbox
    "ExecutionResult",
    "ExecutionTimeoutError",
    "TraceSandbox",
    # Subagent
    "CallBudget",
    "SubAgentConfig",
    "SubAgentLLM",
    "create_ask_llm_function",
    # Trace
    "TraceContext",
    "TraceStep",
]
