"""Prompts and report generation for bullet deduplication."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from ..playbook import Bullet


SIMILARITY_REPORT_HEADER = """
## Similar Bullets Detected

The following bullet pairs have high semantic similarity and may need consolidation.
For each pair, you can decide to:
- **MERGE**: Combine into a single improved bullet (provide merged_content and keep_id)
- **DELETE**: Remove one as redundant (specify bullet_id to delete)
- **KEEP**: Keep both separate if they serve different purposes (explain differentiation)
- **UPDATE**: Refine one bullet's content to clarify the difference (provide new_content)

"""

PAIR_TEMPLATE = """### Pair {index}: {similarity:.0%} similar
**Bullet A** [{id_a}] (helpful={helpful_a}, harmful={harmful_a})
> {content_a}

**Bullet B** [{id_b}] (helpful={helpful_b}, harmful={harmful_b})
> {content_b}

"""


def generate_similarity_report(
    similar_pairs: List[Tuple["Bullet", "Bullet", float]],
) -> str:
    """Generate a human-readable similarity report for the Curator.

    Args:
        similar_pairs: List of (bullet_a, bullet_b, similarity_score) tuples

    Returns:
        Formatted report string to include in Curator prompt
    """
    if not similar_pairs:
        return ""

    parts = [SIMILARITY_REPORT_HEADER]

    for i, (bullet_a, bullet_b, similarity) in enumerate(similar_pairs, 1):
        parts.append(
            PAIR_TEMPLATE.format(
                index=i,
                similarity=similarity,
                id_a=bullet_a.id,
                helpful_a=bullet_a.helpful,
                harmful_a=bullet_a.harmful,
                content_a=bullet_a.content,
                id_b=bullet_b.id,
                helpful_b=bullet_b.helpful,
                harmful_b=bullet_b.harmful,
                content_b=bullet_b.content,
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
      "source_ids": ["bullet-id-1", "bullet-id-2"],
      "keep_id": "bullet-id-1",
      "merged_content": "Improved combined strategy text",
      "reasoning": "Why merging improves the playbook"
    },
    {
      "type": "DELETE",
      "bullet_id": "bullet-id-to-remove",
      "reasoning": "Why this bullet is redundant"
    },
    {
      "type": "KEEP",
      "bullet_ids": ["bullet-id-1", "bullet-id-2"],
      "differentiation": "How they differ in purpose",
      "reasoning": "Why both are needed"
    },
    {
      "type": "UPDATE",
      "bullet_id": "bullet-id-to-update",
      "new_content": "Refined content with context tag like [Batch] or [API]",
      "reasoning": "How this clarifies the distinction"
    }
  ]
}
```

**Guidelines:**
- Consider helpful/harmful counts (higher = more validated, prefer keeping these)
- MERGE when bullets are semantically identical or near-identical
- KEEP when they serve different contexts (batch vs real-time, different APIs, etc.)
- UPDATE to add context tags like "[Batch Jobs]" or "[User-Facing API]" to differentiate
- DELETE only when one is clearly redundant with no unique value

"""
    )

    return "".join(parts)


def format_pair_for_logging(
    bullet_a: "Bullet", bullet_b: "Bullet", similarity: float
) -> str:
    """Format a single pair for logging output."""
    return (
        f"[{bullet_a.id}] '{bullet_a.content[:50]}...' "
        f"↔ [{bullet_b.id}] '{bullet_b.content[:50]}...' "
        f"({similarity:.0%} similar)"
    )
