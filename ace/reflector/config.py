"""Configuration for recursive reflector."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RecursiveConfig:
    """Configuration for the RecursiveReflector.

    Attributes:
        max_iterations: Maximum number of REPL iterations before timing out (default: 15)
        timeout: Timeout in seconds for each code execution (default: 30.0)
        enable_llm_query: Whether to allow llm_query() function in sandbox (default: True)
        max_llm_calls: Maximum number of LLM calls allowed across all functions (default: 30)
        max_context_chars: Maximum total characters in message history before trimming (default: 50000)
        enable_subagent: Whether to enable the ask_llm() sub-agent function (default: True)
        subagent_model: Model to use for sub-agent (None = same as main reflector)
        subagent_max_tokens: Max tokens for sub-agent responses (default: 500)
        subagent_temperature: Temperature for sub-agent responses (default: 0.3)
        subagent_system_prompt: Custom system prompt for sub-agent (None = default)
    """

    max_iterations: int = 15
    timeout: float = 30.0
    enable_llm_query: bool = True
    max_llm_calls: int = 30
    max_context_chars: int = 50_000
    # Sub-agent configuration
    enable_subagent: bool = True
    subagent_model: Optional[str] = None
    subagent_max_tokens: int = 500
    subagent_temperature: float = 0.3
    subagent_system_prompt: Optional[str] = None
