"""Consolidation operations for skill deduplication."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Literal, Union

if TYPE_CHECKING:
    from ..skillbook import Skillbook, SimilarityDecision

logger = logging.getLogger(__name__)


@dataclass
class MergeOp:
    """Merge multiple skills into one.

    Combines helpful/harmful counts from all source skills into the kept skill.
    Other skills are soft-deleted.
    """

    type: Literal["MERGE"] = "MERGE"
    source_ids: List[str] = None  # type: ignore  # All skills being merged
    merged_content: str = ""  # New combined content
    keep_id: str = ""  # Which ID to keep (others deleted)
    reasoning: str = ""

    def __post_init__(self):
        if self.source_ids is None:
            self.source_ids = []


@dataclass
class DeleteOp:
    """Soft-delete a skill as redundant."""

    type: Literal["DELETE"] = "DELETE"
    skill_id: str = ""
    reasoning: str = ""


@dataclass
class KeepOp:
    """Keep both skills separate (they serve different purposes)."""

    type: Literal["KEEP"] = "KEEP"
    skill_ids: List[str] = None  # type: ignore
    differentiation: str = ""  # How they differ
    reasoning: str = ""

    def __post_init__(self):
        if self.skill_ids is None:
            self.skill_ids = []


@dataclass
class UpdateOp:
    """Update a skill's content to differentiate it."""

    type: Literal["UPDATE"] = "UPDATE"
    skill_id: str = ""
    new_content: str = ""
    reasoning: str = ""


# Type alias for any consolidation operation
ConsolidationOperation = Union[MergeOp, DeleteOp, KeepOp, UpdateOp]


def apply_consolidation_operations(
    operations: List[ConsolidationOperation],
    skillbook: "Skillbook",
) -> None:
    """Apply a list of consolidation operations to a skillbook.

    Args:
        operations: List of operations to apply
        skillbook: Skillbook to modify
    """
    for op in operations:
        if isinstance(op, MergeOp):
            _apply_merge(op, skillbook)
        elif isinstance(op, DeleteOp):
            _apply_delete(op, skillbook)
        elif isinstance(op, KeepOp):
            _apply_keep(op, skillbook)
        elif isinstance(op, UpdateOp):
            _apply_update(op, skillbook)
        else:
            logger.warning(f"Unknown operation type: {type(op)}")


def _apply_merge(op: MergeOp, skillbook: "Skillbook") -> None:
    """Apply a MERGE operation."""
    keep_skill = skillbook.get_skill(op.keep_id)
    if keep_skill is None:
        logger.warning(f"MERGE: Keep skill {op.keep_id} not found")
        return

    # Combine metadata from all source skills
    for source_id in op.source_ids:
        if source_id == op.keep_id:
            continue

        source = skillbook.get_skill(source_id)
        if source is None:
            logger.warning(f"MERGE: Source skill {source_id} not found")
            continue

        # Combine counters
        keep_skill.helpful += source.helpful
        keep_skill.harmful += source.harmful
        keep_skill.neutral += source.neutral

        # Soft delete source
        skillbook.remove_skill(source_id, soft=True)
        logger.info(f"MERGE: Soft-deleted {source_id} into {op.keep_id}")

    # Update content to merged version
    if op.merged_content:
        keep_skill.content = op.merged_content

    # Invalidate embedding (needs recomputation)
    keep_skill.embedding = None
    keep_skill.updated_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"MERGE: Completed merge into {op.keep_id}")


def _apply_delete(op: DeleteOp, skillbook: "Skillbook") -> None:
    """Apply a DELETE operation (soft delete)."""
    skill = skillbook.get_skill(op.skill_id)
    if skill is None:
        logger.warning(f"DELETE: Skill {op.skill_id} not found")
        return

    skillbook.remove_skill(op.skill_id, soft=True)
    logger.info(f"DELETE: Soft-deleted {op.skill_id}")


def _apply_keep(op: KeepOp, skillbook: "Skillbook") -> None:
    """Apply a KEEP operation (store decision)."""
    if len(op.skill_ids) < 2:
        logger.warning("KEEP: Need at least 2 skill IDs")
        return

    from ..skillbook import SimilarityDecision

    # Store decision for each pair
    for i, id_a in enumerate(op.skill_ids):
        for id_b in op.skill_ids[i + 1 :]:
            decision = SimilarityDecision(
                decision="KEEP",
                reasoning=op.reasoning or op.differentiation,
                decided_at=datetime.now(timezone.utc).isoformat(),
                similarity_at_decision=0.0,  # We don't have the score here
            )
            skillbook.set_similarity_decision(id_a, id_b, decision)
            logger.info(f"KEEP: Stored decision for ({id_a}, {id_b})")


def _apply_update(op: UpdateOp, skillbook: "Skillbook") -> None:
    """Apply an UPDATE operation."""
    skill = skillbook.get_skill(op.skill_id)
    if skill is None:
        logger.warning(f"UPDATE: Skill {op.skill_id} not found")
        return

    skill.content = op.new_content
    skill.embedding = None  # Needs recomputation
    skill.updated_at = datetime.now(timezone.utc).isoformat()
    logger.info(f"UPDATE: Updated content of {op.skill_id}")
