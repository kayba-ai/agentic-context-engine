"""Shared helpers for ACE role modules."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from ..llm import LLMClient
from ..skillbook import Skillbook

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opik tracing with graceful degradation
# ---------------------------------------------------------------------------
try:
    from ..observability.tracers import maybe_track
except ImportError:
    F = TypeVar("F", bound=Callable[..., Any])

    def maybe_track(
        name: Optional[str] = None, tags: Optional[List[str]] = None, **kwargs: Any
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def _safe_json_loads(text: str) -> Dict[str, Any]:
    # Strip markdown code blocks if present
    text = text.strip()

    # Handle opening fence (with or without language identifier)
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    # Handle closing fence (if present)
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Check if this looks like incomplete JSON (truncated response)
        if "Unterminated string" in str(exc) or "Expecting" in str(exc):
            # Try to detect if this is a truncation issue
            if text.count("{") > text.count("}") or text.rstrip().endswith('"'):
                raise ValueError(
                    f"LLM response appears to be truncated JSON. This may indicate the response was cut off mid-generation. Original error: {exc}\nPartial text: {text[:200]}..."
                ) from exc

        debug_path = Path("logs/json_failures.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write(repr(text))
            fh.write("\n")
        raise ValueError(f"LLM response is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _format_optional(value: Optional[str]) -> str:
    return value or "(none)"


# ---------------------------------------------------------------------------
# Skill citation extraction
# ---------------------------------------------------------------------------
def extract_cited_skill_ids(text: str) -> List[str]:
    """
    Extract skill IDs cited in text using [id-format] notation.

    Parses text to find all skill ID citations in format [section-00001].
    Used to track which strategies were applied by analyzing reasoning traces.

    Args:
        text: Text containing skill citations (reasoning, thoughts, etc.)

    Returns:
        List of unique skill IDs in order of first appearance.
        Empty list if no citations found.

    Example:
        >>> reasoning = "Following [general-00042], I verified the data. Using [geo-00003] for lookup."
        >>> extract_cited_skill_ids(reasoning)
        ['general-00042', 'geo-00003']

    Note:
        Pattern matches: [word_characters-digits]
        Deduplicates while preserving order of first occurrence.
    """
    # Match [section-digits] pattern
    matches = re.findall(r"\[([a-zA-Z_]+-\d+)\]", text)
    # Deduplicate while preserving order
    return list(dict.fromkeys(matches))


# ---------------------------------------------------------------------------
# Skillbook excerpt builder (used by Reflector)
# ---------------------------------------------------------------------------
def _make_skillbook_excerpt(skillbook: Skillbook, skill_ids: Sequence[str]) -> str:
    lines: List[str] = []
    seen = set()
    for skill_id in skill_ids:
        if skill_id in seen:
            continue
        skill = skillbook.get_skill(skill_id)
        if skill:
            seen.add(skill_id)
            lines.append(f"[{skill.id}] {skill.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Instructor auto-wrapping
# ---------------------------------------------------------------------------
def _maybe_wrap_with_instructor(llm: LLMClient, max_retries: int) -> LLMClient:
    """Wrap an LLMClient with Instructor if it doesn't already support structured output."""
    if hasattr(llm, "complete_structured"):
        return llm
    from ..llm_providers.instructor_client import wrap_with_instructor

    return wrap_with_instructor(llm, max_retries=max_retries)  # type: ignore[return-value]
