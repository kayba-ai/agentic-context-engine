"""RR-specific tool registrars and dependency container.

Generic tools (execute_code, recurse) are provided by
:mod:`ace.core.recursive_agent`. This module adds RR-specific
tools and the RR dependency container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import ModelRetry, RunContext

from ace.core.outputs import ReflectorOutput
from ace.core.recursive_agent import AgenticDeps
from ace.core.sandbox import TraceSandbox

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent

from .config import RecursiveConfig


# ------------------------------------------------------------------
# Dependency container
# ------------------------------------------------------------------


@dataclass
class RRDeps(AgenticDeps):
    """Dependencies injected into RR tool calls via ``RunContext``.

    Extends :class:`AgenticDeps` with RR-specific sandbox and trace fields.
    """

    sandbox: TraceSandbox = field(default=None)  # type: ignore[assignment]
    trace_data: dict[str, Any] = field(default_factory=dict)
    skillbook_text: str = ""


# ------------------------------------------------------------------
# RR-specific tool registrars
# ------------------------------------------------------------------


def register_output_validator(agent: "PydanticAgent[RRDeps, Any]") -> None:
    """Register the standard output validator on any RR agent."""

    @agent.output_validator
    def validate_output(
        ctx: RunContext[RRDeps], output: ReflectorOutput
    ) -> ReflectorOutput:
        """Ensure the agent explored data before concluding."""
        if ctx.deps.iteration < 1:
            raise ModelRetry(
                "You haven't explored the data enough. "
                "Use execute_code first, "
                "then provide your final output."
            )
        return output
