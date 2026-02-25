"""ACE integration steps — execute steps for external agentic frameworks.

Each integration provides:

1. **Result type** — integration-specific output (e.g. ``ClaudeCodeResult``)
2. **Execute step** — INJECT + EXECUTE, writes the result to ``ctx.trace``
3. **ToTrace step** — converts the result to a standardised trace dict
   for the learning tail (``ReflectStep``)

Compose with ``learning_tail()``::

    from ace_next.integrations import ClaudeCodeExecuteStep, ClaudeCodeToTrace
    from ace_next.steps import learning_tail

    steps = [
        ClaudeCodeExecuteStep(working_dir="./project"),
        ClaudeCodeToTrace(),
        *learning_tail(reflector, skill_manager, skillbook),
    ]
    pipeline = Pipeline(steps)
"""

from __future__ import annotations

from ..implementations.prompts import wrap_skillbook_for_external_agent

from .browser_use import BrowserExecuteStep, BrowserResult, BrowserToTrace
from .claude_code import ClaudeCodeExecuteStep, ClaudeCodeResult, ClaudeCodeToTrace
from .langchain import LangChainExecuteStep, LangChainResult, LangChainToTrace


def wrap_skillbook_context(skillbook) -> str:
    """Format learned strategies for injection into external agents.

    Thin wrapper around the canonical implementation in
    ``implementations.prompts``.
    """
    return wrap_skillbook_for_external_agent(skillbook)


__all__ = [
    # Browser-use
    "BrowserExecuteStep",
    "BrowserResult",
    "BrowserToTrace",
    # Claude Code
    "ClaudeCodeExecuteStep",
    "ClaudeCodeResult",
    "ClaudeCodeToTrace",
    # LangChain
    "LangChainExecuteStep",
    "LangChainResult",
    "LangChainToTrace",
    # Utility
    "wrap_skillbook_context",
]
