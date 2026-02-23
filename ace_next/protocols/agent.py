"""Protocol defining what steps need from an Agent implementation."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from ..outputs import AgentOutput


@runtime_checkable
class AgentLike(Protocol):
    """Structural interface for Agent-like objects.

    Any object with a matching ``generate`` method satisfies this —
    ``ace.roles.Agent`` and ``ace.roles.ReplayAgent`` both do.
    """

    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        skillbook: Any,
        reflection: Optional[str] = ...,
        **kwargs: Any,
    ) -> AgentOutput: ...
