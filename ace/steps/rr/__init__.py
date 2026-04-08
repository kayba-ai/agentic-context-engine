"""Recursive Reflector as a pipeline step (PydanticAI agent).

"""

from .agent import create_rr_agent
from .config import RecursiveConfig as RRConfig
from .runner import RRStep
from .sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox
from .tools import RRDeps

__all__ = [
    "RRConfig",
    "RRDeps",
    "RRStep",
    "create_rr_agent",
    "ExecutionResult",
    "ExecutionTimeoutError",
    "TraceSandbox",
]
