"""Configuration for recursive reflector."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RecursiveConfig:
    """Configuration for the RecursiveReflector.

    Attributes:
        max_iterations: Maximum number of REPL iterations before timing out (default: 10)
        timeout: Timeout in seconds for each code execution (default: 30.0)
        enable_llm_query: Whether to allow llm_query() function in sandbox (default: True)
        max_llm_calls: Maximum number of llm_query() calls allowed (default: 20)
        fallback_on_error: Whether to fall back to simple reflection on error (default: True)
        code_block_marker: Marker to detect code blocks in LLM response (default: "```python")
        additional_imports: Extra modules/variables to inject into sandbox
        enable_subagent: Whether to enable the ask_llm() sub-agent function (default: True)
        subagent_model: Model to use for sub-agent (None = same as main reflector)
        subagent_max_tokens: Max tokens for sub-agent responses (default: 500)
        subagent_temperature: Temperature for sub-agent responses (default: 0.3)
        subagent_system_prompt: Custom system prompt for sub-agent (None = default)
    """

    max_iterations: int = 10
    timeout: float = 30.0
    enable_llm_query: bool = True
    max_llm_calls: int = 20
    fallback_on_error: bool = True
    code_block_marker: str = "```python"
    additional_imports: Dict[str, Any] = field(default_factory=dict)
    # Sub-agent configuration
    enable_subagent: bool = True
    subagent_model: Optional[str] = None
    subagent_max_tokens: int = 500
    subagent_temperature: float = 0.3
    subagent_system_prompt: Optional[str] = None
