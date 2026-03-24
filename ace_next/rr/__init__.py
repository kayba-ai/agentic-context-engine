"""Recursive Reflector as a pipeline step (PydanticAI agent).

Public API::

    from ace_next.rr import RRStep, RRConfig

    rr = RRStep("gpt-4o-mini", config=RRConfig(max_llm_calls=30))
    pipe = Pipeline([..., rr, ...])
"""

from .agent import RRDeps, create_rr_agent, create_sub_agent
from .config import RecursiveConfig as RRConfig
from .runner import RRStep
from .sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox
from .trace_context import TraceContext, TraceStep


def __getattr__(name: str):
    """Lazy-import optional components."""
    if name == "RROpikStep":
        from .opik import RROpikStep
        return RROpikStep
    # Backward compat — old inner pipeline steps (deprecated)
    _deprecated = {
        "RRIterationContext": ".context",
        "CheckResultStep": ".steps",
        "ExtractCodeStep": ".steps",
        "LLMCallStep": ".steps",
        "SandboxExecStep": ".steps",
        "CallBudget": ".subagent",
        "SubAgentConfig": ".subagent",
        "SubAgentLLM": ".subagent",
        "create_ask_llm_function": ".subagent",
    }
    if name in _deprecated:
        import importlib
        import warnings
        warnings.warn(
            f"{name} is deprecated. The RR now uses PydanticAI agents internally. "
            "See docs/PYDANTIC_AI_MIGRATION.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(_deprecated[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RRConfig",
    "RRDeps",
    "RROpikStep",
    "RRStep",
    # Agent factories
    "create_rr_agent",
    "create_sub_agent",
    # Sandbox
    "ExecutionResult",
    "ExecutionTimeoutError",
    "TraceSandbox",
    # Trace
    "TraceContext",
    "TraceStep",
]
