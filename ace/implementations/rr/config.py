"""Configuration for recursive reflector."""

from dataclasses import dataclass

from ...core.recursive_agent import AgenticConfig


@dataclass
class RecursiveConfig(AgenticConfig):
    """Configuration for the RecursiveReflector.

    Extends :class:`AgenticConfig` with RR-specific fields for sandbox
    execution and output limits.

    Attributes:
        timeout: Timeout in seconds for each code execution (default: 30.0)
        max_output_chars: Maximum characters per code execution output before truncation (default: 20000)
    """

    timeout: float = 30.0
    max_output_chars: int = 20_000
