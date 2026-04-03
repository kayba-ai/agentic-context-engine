"""Recursive Reflector as a pipeline step (PydanticAI agent).

Public API::

    from ace.rr import RRStep, RRConfig

    rr = RRStep("gpt-4o-mini", config=RRConfig(max_llm_calls=30))
    pipe = Pipeline([..., rr, ...])
"""

from .agent import (
    RRDeps,
    create_orchestrator_agent,
    create_rr_agent,
    create_sub_agent,
    create_worker_agent,
)
from .config import RecursiveConfig as RRConfig
from .runner import RRStep
from .sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox
from .trace_context import TraceContext, TraceStep

__all__ = [
    "RRConfig",
    "RRDeps",
    "RRStep",
    # Agent factories
    "create_orchestrator_agent",
    "create_rr_agent",
    "create_sub_agent",
    "create_worker_agent",
    # Sandbox
    "ExecutionResult",
    "ExecutionTimeoutError",
    "TraceSandbox",
    # Trace
    "TraceContext",
    "TraceStep",
]
