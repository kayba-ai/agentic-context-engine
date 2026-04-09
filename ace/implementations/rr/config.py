"""Configuration for recursive reflector."""

from dataclasses import dataclass

from ...core.recursive_agent import AgenticConfig


@dataclass
class RecursiveConfig(AgenticConfig):
    """Configuration for the Recursive Reflector.

    Inherits all fields from :class:`AgenticConfig`. Overrides
    ``max_output_chars`` for larger trace outputs.
    """

    max_output_chars: int = 50_000
