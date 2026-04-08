"""Recursive Reflector as a pipeline step (PydanticAI agent).

"""

from .agent import (
    create_orchestrator_agent,
    create_rr_agent,
    create_sub_agent,
    create_worker_agent,
)
from .config import RecursiveConfig as RRConfig
from .runner import RRStep
from .sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox
from .tools import RRDeps

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
]
