"""Consolidation operations for bullet deduplication."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Literal, Union

if TYPE_CHECKING:
    from ..playbook import Playbook, SimilarityDecision

logger = logging.getLogger(__name__)


@dataclass
class MergeOp:
    """Merge multiple bullets into one.

    Combines helpful/harmful counts from all source bullets into the kept bullet.
    Other bullets are soft-deleted.
    """

    type: Literal["MERGE"] = "MERGE"
    source_ids: List[str] = None  # type: ignore  # All bullets being merged
    merged_content: str = ""  # New combined content
    keep_id: str = ""  # Which ID to keep (others deleted)
    reasoning: str = ""

    def __post_init__(self):
        if self.source_ids is None:
            self.source_ids = []


@dataclass
class DeleteOp:
    """Soft-delete a bullet as redundant."""

    type: Literal["DELETE"] = "DELETE"
    bullet_id: str = ""
    reasoning: str = ""


@dataclass
class KeepOp:
    """Keep both bullets separate (they serve different purposes)."""

    type: Literal["KEEP"] = "KEEP"
    bullet_ids: List[str] = None  # type: ignore
    differentiation: str = ""  # How they differ
    reasoning: str = ""

    def __post_init__(self):
        if self.bullet_ids is None:
            self.bullet_ids = []


@dataclass
class UpdateOp:
    """Update a bullet's content to differentiate it."""

    type: Literal["UPDATE"] = "UPDATE"
    bullet_id: str = ""
    new_content: str = ""
    reasoning: str = ""


# Type alias for any consolidation operation
ConsolidationOperation = Union[MergeOp, DeleteOp, KeepOp, UpdateOp]


def apply_consolidation_operations(
    operations: List[ConsolidationOperation],
    playbook: "Playbook",
) -> None:
    """Apply a list of consolidation operations to a playbook.

    Args:
        operations: List of operations to apply
        playbook: Playbook to modify
    """
    for op in operations:
        if isinstance(op, MergeOp):
            _apply_merge(op, playbook)
        elif isinstance(op, DeleteOp):
            _apply_delete(op, playbook)
        elif isinstance(op, KeepOp):
            _apply_keep(op, playbook)
        elif isinstance(op, UpdateOp):
            _apply_update(op, playbook)
        else:
            logger.warning(f"Unknown operation type: {type(op)}")


def _apply_merge(op: MergeOp, playbook: "Playbook") -> None:
    """Apply a MERGE operation."""
    keep_bullet = playbook.get_bullet(op.keep_id)
    if keep_bullet is None:
        logger.warning(f"MERGE: Keep bullet {op.keep_id} not found")
        return

    # Combine metadata from all source bullets
    for source_id in op.source_ids:
        if source_id == op.keep_id:
            continue

        source = playbook.get_bullet(source_id)
        if source is None:
            logger.warning(f"MERGE: Source bullet {source_id} not found")
            continue

        # Combine counters
        keep_bullet.helpful += source.helpful
        keep_bullet.harmful += source.harmful
        keep_bullet.neutral += source.neutral

        # Soft delete source
        playbook.remove_bullet(source_id, soft=True)
        logger.info(f"MERGE: Soft-deleted {source_id} into {op.keep_id}")

    # Update content to merged version
    if op.merged_content:
        keep_bullet.content = op.merged_content

    # Invalidate embedding (needs recomputation)
    keep_bullet.embedding = None
    keep_bullet.updated_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"MERGE: Completed merge into {op.keep_id}")


def _apply_delete(op: DeleteOp, playbook: "Playbook") -> None:
    """Apply a DELETE operation (soft delete)."""
    bullet = playbook.get_bullet(op.bullet_id)
    if bullet is None:
        logger.warning(f"DELETE: Bullet {op.bullet_id} not found")
        return

    playbook.remove_bullet(op.bullet_id, soft=True)
    logger.info(f"DELETE: Soft-deleted {op.bullet_id}")


def _apply_keep(op: KeepOp, playbook: "Playbook") -> None:
    """Apply a KEEP operation (store decision)."""
    if len(op.bullet_ids) < 2:
        logger.warning("KEEP: Need at least 2 bullet IDs")
        return

    from ..playbook import SimilarityDecision

    # Store decision for each pair
    for i, id_a in enumerate(op.bullet_ids):
        for id_b in op.bullet_ids[i + 1 :]:
            decision = SimilarityDecision(
                decision="KEEP",
                reasoning=op.reasoning or op.differentiation,
                decided_at=datetime.now(timezone.utc).isoformat(),
                similarity_at_decision=0.0,  # We don't have the score here
            )
            playbook.set_similarity_decision(id_a, id_b, decision)
            logger.info(f"KEEP: Stored decision for ({id_a}, {id_b})")


def _apply_update(op: UpdateOp, playbook: "Playbook") -> None:
    """Apply an UPDATE operation."""
    bullet = playbook.get_bullet(op.bullet_id)
    if bullet is None:
        logger.warning(f"UPDATE: Bullet {op.bullet_id} not found")
        return

    bullet.content = op.new_content
    bullet.embedding = None  # Needs recomputation
    bullet.updated_at = datetime.now(timezone.utc).isoformat()
    logger.info(f"UPDATE: Updated content of {op.bullet_id}")
