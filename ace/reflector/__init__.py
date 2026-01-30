"""Recursive reflector module for ACE framework.

This module provides a recursive reflector that uses code execution
to analyze agent traces more thoroughly than single-pass reflection.

Example:
    >>> from ace.reflector import RecursiveReflector, RecursiveConfig
    >>> from ace.llm_providers.litellm_client import LiteLLMClient
    >>>
    >>> llm = LiteLLMClient(model="gpt-4o-mini")
    >>> reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=5))
"""

from .config import RecursiveConfig
from .sandbox import TraceSandbox, ExecutionResult, ExecutionTimeoutError
from .trace_context import TraceContext, TraceStep
from .recursive import RecursiveReflector
from .prompts import REFLECTOR_RECURSIVE_PROMPT, REFLECTOR_RECURSIVE_SYSTEM

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
    # Prompts
    "REFLECTOR_RECURSIVE_PROMPT",
    "REFLECTOR_RECURSIVE_SYSTEM",
]
