"""Configuration for recursive reflector."""

from dataclasses import dataclass, field
from typing import Any, Dict


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
    """

    max_iterations: int = 10
    timeout: float = 30.0
    enable_llm_query: bool = True
    max_llm_calls: int = 20
    fallback_on_error: bool = True
    code_block_marker: str = "```python"
    additional_imports: Dict[str, Any] = field(default_factory=dict)
