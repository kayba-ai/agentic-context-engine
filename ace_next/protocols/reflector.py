"""Protocol defining what steps need from a Reflector implementation."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from ..outputs import AgentOutput, ReflectorOutput


@runtime_checkable
class ReflectorLike(Protocol):
    """Structural interface for Reflector-like objects.

    Any object with a matching ``reflect`` method satisfies this —
    ``ace.roles.Reflector`` does.
    """

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any,
        ground_truth: Optional[str] = ...,
        feedback: Optional[str] = ...,
        **kwargs: Any,
    ) -> ReflectorOutput: ...
