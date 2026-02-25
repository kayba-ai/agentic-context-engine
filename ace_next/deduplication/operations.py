"""Consolidation operations for skill deduplication."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Literal, Union

if TYPE_CHECKING:
    from ..core.skillbook import Skillbook

logger = logging.getLogger(__name__)


@dataclass
class MergeOp:
    """Merge multiple skills into one.

    Combines helpful/harmful counts from all source skills into the kept skill.
    Other skills are soft-deleted.
    """

    type: Literal["MERGE"] = "MERGE"
    source_ids: List[str] = None  # type: ignore[assignment]
    merged_content: str = ""
    keep_id: str = ""
    reasoning: str = ""

    def __post_init__(self) -> None:
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
    skill_ids: List[str] = None  # type: ignore[assignment]
    differentiation: str = ""
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.skill_ids is None:
            self.skill_ids = []


@dataclass
class UpdateOp:
    """Update a skill's content to differentiate it."""

    type: Literal["UPDATE"] = "UPDATE"
    skill_id: str = ""
    new_content: str = ""
    reasoning: str = ""


ConsolidationOperation = Union[MergeOp, DeleteOp, KeepOp, UpdateOp]


# ---------------------------------------------------------------------------
# Apply helpers
# ---------------------------------------------------------------------------


def apply_consolidation_operations(
    operations: List[ConsolidationOperation],
    skillbook: "Skillbook",
) -> None:
    """Apply a list of consolidation operations to a skillbook."""
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
            logger.warning("Unknown operation type: %s", type(op))


def _apply_merge(op: MergeOp, skillbook: "Skillbook") -> None:
    keep_skill = skillbook.get_skill(op.keep_id)
    if keep_skill is None:
        logger.warning("MERGE: Keep skill %s not found", op.keep_id)
        return

    for source_id in op.source_ids:
        if source_id == op.keep_id:
            continue
        source = skillbook.get_skill(source_id)
        if source is None:
            logger.warning("MERGE: Source skill %s not found", source_id)
            continue
        keep_skill.helpful += source.helpful
        keep_skill.harmful += source.harmful
        keep_skill.neutral += source.neutral
        skillbook.remove_skill(source_id, soft=True)
        logger.info("MERGE: Soft-deleted %s into %s", source_id, op.keep_id)

    if op.merged_content:
        keep_skill.content = op.merged_content

    keep_skill.embedding = None
    keep_skill.updated_at = datetime.now(timezone.utc).isoformat()
    logger.info("MERGE: Completed merge into %s", op.keep_id)


def _apply_delete(op: DeleteOp, skillbook: "Skillbook") -> None:
    skill = skillbook.get_skill(op.skill_id)
    if skill is None:
        logger.warning("DELETE: Skill %s not found", op.skill_id)
        return
    skillbook.remove_skill(op.skill_id, soft=True)
    logger.info("DELETE: Soft-deleted %s", op.skill_id)


def _apply_keep(op: KeepOp, skillbook: "Skillbook") -> None:
    if len(op.skill_ids) < 2:
        logger.warning("KEEP: Need at least 2 skill IDs")
        return

    from ..core.skillbook import SimilarityDecision

    for i, id_a in enumerate(op.skill_ids):
        for id_b in op.skill_ids[i + 1 :]:
            decision = SimilarityDecision(
                decision="KEEP",
                reasoning=op.reasoning or op.differentiation,
                decided_at=datetime.now(timezone.utc).isoformat(),
                similarity_at_decision=0.0,
            )
            skillbook.set_similarity_decision(id_a, id_b, decision)
            logger.info("KEEP: Stored decision for (%s, %s)", id_a, id_b)


def _apply_update(op: UpdateOp, skillbook: "Skillbook") -> None:
    skill = skillbook.get_skill(op.skill_id)
    if skill is None:
        logger.warning("UPDATE: Skill %s not found", op.skill_id)
        return
    skill.content = op.new_content
    skill.embedding = None
    skill.updated_at = datetime.now(timezone.utc).isoformat()
    logger.info("UPDATE: Updated content of %s", op.skill_id)
