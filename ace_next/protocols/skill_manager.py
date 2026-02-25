"""Protocol defining what steps need from a SkillManager implementation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.outputs import ReflectorOutput, SkillManagerOutput


@runtime_checkable
class SkillManagerLike(Protocol):
    """Structural interface for SkillManager-like objects.

    Any object with a matching ``update_skills`` method satisfies this —
    ``ace.roles.SkillManager`` does.
    """

    def update_skills(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput: ...
