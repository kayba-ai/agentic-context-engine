"""Shared utilities for ACE role implementations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    from ..skillbook import Skillbook


def extract_cited_skill_ids(text: str) -> List[str]:
    """Extract skill IDs cited in text using ``[id-format]`` notation.

    Parses ``[section-00001]`` patterns and returns unique IDs in order
    of first appearance.

    Args:
        text: Text containing skill citations.

    Returns:
        Deduplicated list of skill IDs preserving first-occurrence order.

    Example::

        >>> extract_cited_skill_ids("Following [general-00042], I verified the data.")
        ['general-00042']
    """
    matches = re.findall(r"\[([a-zA-Z_]+-\d+)\]", text)
    return list(dict.fromkeys(matches))


def format_optional(value: Optional[str]) -> str:
    """Return *value* or ``"(none)"`` when falsy."""
    return value or "(none)"


def make_skillbook_excerpt(
    skillbook: "Skillbook", skill_ids: Sequence[str]
) -> str:
    """Build a compact excerpt of cited skills.

    Args:
        skillbook: Skillbook to look up skills in.
        skill_ids: Ordered skill IDs cited by the agent.

    Returns:
        One ``[id] content`` line per unique cited skill found.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for skill_id in skill_ids:
        if skill_id in seen:
            continue
        skill = skillbook.get_skill(skill_id)
        if skill:
            seen.add(skill_id)
            lines.append(f"[{skill.id}] {skill.content}")
    return "\n".join(lines)
