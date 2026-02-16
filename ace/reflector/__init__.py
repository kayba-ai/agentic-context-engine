"""Recursive reflector module for ACE framework.

This module provides a recursive reflector that uses code execution
to analyze agent traces more thoroughly than single-pass reflection.

Example:
    >>> from ace.reflector import RecursiveReflector, RecursiveConfig
    >>> from ace.llm_providers.litellm_client import LiteLLMClient
    >>>
    >>> llm = LiteLLMClient(model="gpt-4o-mini")
    >>> reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=5))

The module also provides a sub-agent LLM wrapper for trace exploration:

    >>> from ace.reflector import SubAgentConfig, create_ask_llm_function
    >>> ask_llm = create_ask_llm_function(llm, max_calls=10)
    >>> insight = ask_llm("What went wrong?", context="Error: timeout")
"""

from .config import RecursiveConfig
from .sandbox import TraceSandbox, ExecutionResult, ExecutionTimeoutError
from .trace_context import TraceContext, TraceStep
from .recursive import RecursiveReflector
from .prompts import REFLECTOR_RECURSIVE_PROMPT, REFLECTOR_RECURSIVE_SYSTEM
from .prompts_rr_v3 import REFLECTOR_RECURSIVE_V3_PROMPT, REFLECTOR_RECURSIVE_V3_SYSTEM
from .subagent import SubAgentConfig, SubAgentLLM, create_ask_llm_function

__all__ = [
    # Config
    "RecursiveConfig",
    # Sandbox
    "TraceSandbox",
    "ExecutionResult",
    "ExecutionTimeoutError",
    # Trace
    "TraceContext",
    "TraceStep",
    # Reflector
    "RecursiveReflector",
    # Sub-agent
    "SubAgentConfig",
    "SubAgentLLM",
    "create_ask_llm_function",
    # Prompts (v2 kept for backward compat)
    "REFLECTOR_RECURSIVE_PROMPT",
    "REFLECTOR_RECURSIVE_SYSTEM",
    # Prompts v3 (default)
    "REFLECTOR_RECURSIVE_V3_PROMPT",
    "REFLECTOR_RECURSIVE_V3_SYSTEM",
]
