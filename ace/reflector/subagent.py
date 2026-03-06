"""Backward-compat shim — canonical location is ace_next.rr.subagent."""

from ace_next.rr.subagent import (
    CallBudget,
    SubAgentConfig,
    SubAgentLLM,
    create_ask_llm_function,
    DEFAULT_SUBAGENT_SYSTEM_PROMPT,
    SUBAGENT_ANALYSIS_PROMPT,
    SUBAGENT_DEEPDIVE_PROMPT,
)

__all__ = [
    "CallBudget",
    "SubAgentConfig",
    "SubAgentLLM",
    "create_ask_llm_function",
    "DEFAULT_SUBAGENT_SYSTEM_PROMPT",
    "SUBAGENT_ANALYSIS_PROMPT",
    "SUBAGENT_DEEPDIVE_PROMPT",
]
