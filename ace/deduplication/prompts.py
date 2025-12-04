"""Prompts and report generation for skill deduplication."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from ..skillbook import Skill


SIMILARITY_REPORT_HEADER = """
## Similar Skills Detected

The following skill pairs have high semantic similarity and may need consolidation.
For each pair, you can decide to:
- **MERGE**: Combine into a single improved skill (provide merged_content and keep_id)
- **DELETE**: Remove one as redundant (specify skill_id to delete)
- **KEEP**: Keep both separate if they serve different purposes (explain differentiation)
- **UPDATE**: Refine one skill's content to clarify the difference (provide new_content)

"""

PAIR_TEMPLATE = """### Pair {index}: {similarity:.0%} similar
**Skill A** [{id_a}] (helpful={helpful_a}, harmful={harmful_a})
> {content_a}

**Skill B** [{id_b}] (helpful={helpful_b}, harmful={harmful_b})
> {content_b}

"""


def generate_similarity_report(
    similar_pairs: List[Tuple["Skill", "Skill", float]],
) -> str:
    """Generate a human-readable similarity report for the SkillManager.

    Args:
        similar_pairs: List of (skill_a, skill_b, similarity_score) tuples

    Returns:
        Formatted report string to include in SkillManager prompt
    """
    if not similar_pairs:
        return ""

    parts = [SIMILARITY_REPORT_HEADER]

    for i, (skill_a, skill_b, similarity) in enumerate(similar_pairs, 1):
        parts.append(
            PAIR_TEMPLATE.format(
                index=i,
                similarity=similarity,
                id_a=skill_a.id,
                helpful_a=skill_a.helpful,
                harmful_a=skill_a.harmful,
                content_a=skill_a.content,
                id_b=skill_b.id,
                helpful_b=skill_b.helpful,
                harmful_b=skill_b.harmful,
                content_b=skill_b.content,
            )
        )

    parts.append(
        """
## Consolidation Operations Format

Include consolidation operations in your response under a `consolidation_operations` key.
Each operation should have a `type` field and relevant fields for that type:

```json
{
  "consolidation_operations": [
    {
      "type": "MERGE",
      "source_ids": ["skill-id-1", "skill-id-2"],
      "keep_id": "skill-id-1",
      "merged_content": "Improved combined strategy text",
      "reasoning": "Why merging improves the skillbook"
    },
    {
      "type": "DELETE",
      "skill_id": "skill-id-to-remove",
      "reasoning": "Why this skill is redundant"
    },
    {
      "type": "KEEP",
      "skill_ids": ["skill-id-1", "skill-id-2"],
      "differentiation": "How they differ in purpose",
      "reasoning": "Why both are needed"
    },
    {
      "type": "UPDATE",
      "skill_id": "skill-id-to-update",
      "new_content": "Refined content with context tag like [Batch] or [API]",
      "reasoning": "How this clarifies the distinction"
    }
  ]
}
```

**Guidelines:**
- Consider helpful/harmful counts (higher = more validated, prefer keeping these)
- MERGE when skills are semantically identical or near-identical
- KEEP when they serve different contexts (batch vs real-time, different APIs, etc.)
- UPDATE to add context tags like "[Batch Jobs]" or "[User-Facing API]" to differentiate
- DELETE only when one is clearly redundant with no unique value

"""
    )

    return "".join(parts)


def format_pair_for_logging(
    skill_a: "Skill", skill_b: "Skill", similarity: float
) -> str:
    """Format a single pair for logging output."""
    return (
        f"[{skill_a.id}] '{skill_a.content[:50]}...' "
        f"↔ [{skill_b.id}] '{skill_b.content[:50]}...' "
        f"({similarity:.0%} similar)"
    )
