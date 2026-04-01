"""XML skill rendering and per-task skill retrieval.

Provides an alternative to the TOON-based ``Skillbook.as_prompt()`` that
renders skills as XML ``<strategy>`` elements and supports embedding-based
top-k retrieval for per-task skill injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ace.core.skillbook import Skill, Skillbook
    from ace.deduplication.detector import SimilarityDetector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# XML rendering
# ------------------------------------------------------------------

def render_skills_xml(skills: list[Skill]) -> str:
    """Render a list of skills as XML ``<strategy>`` elements.

    Each skill becomes::

        <strategy id="general-00042" section="general">
        When a customer requests a flight change, ...
        </strategy>

    Args:
        skills: Skills to render.

    Returns:
        XML string with all skills, or empty string if no skills.
    """
    if not skills:
        return ""

    parts: list[str] = []
    for s in skills:
        parts.append(
            f'<strategy id="{s.id}" section="{s.section}">\n'
            f"{s.content}\n"
            f"</strategy>"
        )

    strategies_block = "\n".join(parts)

    return (
        f"{strategies_block}\n\n"
        "Adapt these strategies to your current situation — "
        "they are patterns, not rigid rules."
    )


# ------------------------------------------------------------------
# Per-task retrieval
# ------------------------------------------------------------------

def retrieve_top_k(
    skillbook: Skillbook,
    query: str,
    *,
    top_k: int = 5,
    detector: SimilarityDetector | None = None,
) -> list[Skill]:
    """Retrieve the most relevant skills for a query via embedding similarity.

    Uses the existing ``SimilarityDetector`` infrastructure to embed the
    query and compute cosine similarity against all active skill embeddings.

    Args:
        skillbook: Skillbook with active skills (embeddings computed lazily).
        query: Task description or user scenario text.
        top_k: Number of skills to return.
        detector: Pre-initialized detector (created with defaults if None).

    Returns:
        Top-k skills sorted by descending similarity.
    """
    if detector is None:
        from ace.deduplication.detector import SimilarityDetector as _Det
        from ace.protocols.deduplication import DeduplicationConfig

        detector = _Det(DeduplicationConfig())

    # Ensure all skills have embeddings (idempotent)
    detector.ensure_embeddings(skillbook)

    # Embed the query
    query_embedding = detector.compute_embedding(query)
    if query_embedding is None:
        logger.warning("Failed to embed query; returning all active skills")
        return skillbook.skills()[:top_k]

    # Score each skill
    scored: list[tuple[float, Skill]] = []
    for skill in skillbook.skills():
        if skill.embedding is None:
            continue
        sim = detector.cosine_similarity(query_embedding, skill.embedding)
        scored.append((sim, skill))

    # Sort descending by similarity
    scored.sort(key=lambda x: x[0], reverse=True)

    return [skill for _, skill in scored[:top_k]]
